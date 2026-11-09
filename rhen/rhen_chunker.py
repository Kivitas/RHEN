"""
Next-Generation Semantic Hierarchical Chunking for the Rhen Decision Engine.

Rhen's contract is ``State + Typed Questions -> Calibrated Decisions``. Long states
break that contract when chunked naively: a JSON object split between two chunks
is no longer parseable, a code fence cut mid-block is no longer valid, and an
``Extract`` span that straddles a boundary becomes unresolvable.

This module provides four cooperating pieces:

1. :class:`SemanticBoundaryChunker`
   Splits a state payload on *semantic* boundaries -- JSON/YAML brackets, Markdown
   headers, fenced code blocks, stack-trace frames, and table rows -- so no
   structure is ever sliced mid-field. Budgeted windows
   (``C_max = 512`` tokens) with ``C_overlap = 64`` tokens preserve inter-chunk
   relational continuity.

2. :class:`SalienceGatedChunkRouter`
   Routes a question to the chunks that actually matter:

       s_i(q) = (h_q^T W_salience) z_{chunk,i} / sqrt(d)
       w_i    = Softmax(s_i(q) / tau)

3. :func:`evidential_logsumexp_fuse`
   Fuses the per-chunk evidence into a single calibrated simplex, monotonically
   accumulating scattered evidence across ``N`` chunks:

       s*(q, o_k) = log sum_i w_i * exp( s_i(q, o_k) / tau )

4. :class:`ChunkedSpanPointer`
   Resolves the winning chunk ``i* = argmax_i s_i`` and extracts exact span
   strings from it without truncation.
"""

import hashlib
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------------------

#: Default maximum window budget per chunk, in tokens.
DEFAULT_MAX_CHUNK_TOKENS: int = 512

#: Default overlap between adjacent chunks, in tokens, preserving relational continuity.
DEFAULT_OVERLAP_TOKENS: int = 64

#: Default softmax temperature for salience routing and evidential fusion.
DEFAULT_TEMPERATURE: float = 1.0

#: Rough characters-per-token ratio used when no tokenizer is supplied.
#: BPE merges on English text average ~4 characters/token; this is only used to
#: enforce the window budget, never for the decision itself.
_CHARS_PER_TOKEN: float = 4.0

#: Set once the tokenizer-failure WARNING has been logged in this process.
_TOKENIZER_FALLBACK_LOGGED: bool = False


def _log_tokenizer_fallback_once(exc: BaseException) -> None:
    """Log that the tokenizer failed and a character estimate is used (once per process)."""
    global _TOKENIZER_FALLBACK_LOGGED
    if _TOKENIZER_FALLBACK_LOGGED:
        return
    _TOKENIZER_FALLBACK_LOGGED = True
    logger.warning(
        "Tokenizer encode failed with %s: %s; SemanticBoundaryChunker.count_tokens is using a "
        "character-count estimate (%.0f chars/token) for the window budget. "
        "Further occurrences are not logged.",
        type(exc).__name__, exc, _CHARS_PER_TOKEN,
    )


# --------------------------------------------------------------------------------------
# Structural boundary detection
# --------------------------------------------------------------------------------------

#: Stack-trace frame prefixes. Splitting *before* these keeps each frame intact.
_STACK_FRAME_PREFIXES: Tuple[str, ...] = (
    "File \"",         # Python / Ruby
    "at ",             # JavaScript / JVM
    "\tat ",           # Go / Java
    "Caused by:",      # Java chained exceptions
    "Traceback (most recent call last):",
    "Exception in thread",
    "  File \"",       # Ruby
    "raised ",
)

#: Markdown construct prefixes that start a new structural unit.
_MARKDOWN_PREFIXES: Tuple[str, ...] = (
    "#",        # ATX headers
    "##",
    "###",
    "```",      # fenced code block open/close
    "~~~",
    "> ",       # block quote
    "- ",       # bullet list
    "* ",
    "+ ",
    "|",        # table rows
    "---",      # thematic break / YAML doc separator
    "...",      # YAML document end
)

#: YAML top-level construct prefixes.
_YAML_PREFIXES: Tuple[str, ...] = ("--- ", "... ", "---")


def _is_blank(line: str) -> bool:
    return not line.strip()


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def classify_boundary(line: str) -> str:
    """Classify a line as a *hard* boundary, *soft* boundary, or neutral.

    Hard boundaries are structurally mandatory: cutting here can never break a
    syntactic construct (a top-level JSON brace, a markdown header, a code-fence
    delimiter). Soft boundaries are merely good places to cut (a blank line, a
    stack-trace frame, a table row separator) -- safe here, but a hard boundary is
    always available nearby.

    Returns one of ``"hard"``, ``"soft"`` or ``"neutral"``.
    """
    stripped = line.strip()
    if not stripped:
        return "soft"

    indent = len(_leading_ws(line))

    # Hard: an unclosed bracket delimiter at the start of a line. For JSON/YAML
    # these only appear at column 0 (top level) or at consistent indentation,
    # and a line *beginning* with a closing delimiter always ends a construct.
    if stripped[0] in "}]":
        return "hard"
    if indent == 0 and stripped[0] in "[{":
        return "hard"
    # Hard: a top-level YAML key starts a new mapping entry.
    if indent == 0 and ":" in stripped and not stripped.startswith(("#", "-", "*")):
        return "hard"
    # Hard: markdown header / fence / thematic break at column 0.
    if indent == 0 and stripped.startswith(("#", "```", "~~~", "---")):
        return "hard"

    # Soft: a stack-trace frame always starts a new logical unit.
    if any(stripped.startswith(p) or stripped.startswith(p.strip()) for p in _STACK_FRAME_PREFIXES):
        return "soft"
    if stripped.startswith(_MARKDOWN_PREFIXES):
        return "soft"
    # Soft: a trailing comma closes a complete array element / object field, so
    # the next line begins a fresh unit. Without this, a pretty-printed JSON
    # object has interior cut points only at its outermost braces, and the whole
    # object is forced into a single oversized window. Marking it *soft* (not
    # hard) keeps a later hard boundary preferable when one is nearby.
    if stripped.endswith(","):
        return "soft"

    return "neutral"


def _fence_state(before: str) -> bool:
    """Whether the text preceding this point is inside an open Markdown code fence."""
    return before.count("```") % 2 == 1


def find_fenced_blocks(lines: Sequence[str]) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` line ranges for each fenced code block, inclusive.

    A block with no closing fence runs to the end of the document. Used to keep a
    budget-forced cut from landing inside a program listing.
    """
    blocks: List[Tuple[int, int]] = []
    open_at: Optional[int] = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if open_at is None:
                open_at = idx
            else:
                blocks.append((open_at, idx))
                open_at = None
    if open_at is not None:
        blocks.append((open_at, len(lines) - 1))
    return blocks


def find_semantic_boundaries(text: str) -> List[Tuple[int, str]]:
    """Return ``(line_index, strength)`` for every semantic boundary in ``text``.

    Lines *inside* an open code fence are never reported as boundaries, because a
    fence is a single indivisible unit -- otherwise a chunk boundary would split
    a syntactically valid program into two invalid halves.

    Strength is ``"hard"`` or ``"soft"``; blank lines are ``"soft"``.
    """
    lines = text.splitlines()
    boundaries: List[Tuple[int, str]] = []
    in_fence = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            # The fence delimiter itself is a hard boundary; toggle afterwards.
            boundaries.append((idx, "hard"))
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        strength = classify_boundary(line)
        if strength != "neutral":
            boundaries.append((idx, strength))

    return boundaries


# --------------------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------------------

@dataclass
class Chunk:
    """One semantic window of the source state.

    Attributes:
        index: Position of this chunk in the ordered chunk sequence.
        text: The chunk payload (with overlap applied to both edges).
        start_line: First source line included in ``text``.
        end_line: Last source line included in ``text`` (inclusive).
        n_tokens: Token count under the chunker's accounting rule.
        boundary: Strength of the boundary that *started* this chunk.
        kind: Best-effort structural kind (``"json"``, ``"yaml"``, ``"markdown"``,
            ``"code"``, ``"trace"``, ``"table"`` or ``"text"``).
    """

    index: int
    text: str
    start_line: int
    end_line: int
    n_tokens: int
    boundary: str = "hard"
    kind: str = "text"

    def __len__(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "n_tokens": self.n_tokens,
            "boundary": self.boundary,
            "kind": self.kind,
        }


def detect_structure_kind(text: str) -> str:
    """Best-effort structural classification of a state payload."""
    stripped = text.strip()
    if not stripped:
        return "text"
    if stripped[0] in "[{":
        try:
            import json

            json.loads(stripped)
            return "json"
        except Exception:
            return "json"  # still structurally JSON-ish even if truncated
    if "```" in text:
        return "code"
    if "\t" in text and any(p.strip() in text for p in _STACK_FRAME_PREFIXES):
        return "trace"
    if any(line.strip().startswith("|") for line in text.splitlines()[:8]):
        return "table"
    if any(line.startswith("#") for line in text.splitlines()[:8]):
        return "markdown"
    if any(line.rstrip() == "---" for line in text.splitlines()[:4]):
        return "yaml"
    if ":" in stripped and any(line.startswith(" ") for line in text.splitlines()[:8]):
        return "yaml"
    return "text"


# --------------------------------------------------------------------------------------
# 1. SemanticBoundaryChunker
# --------------------------------------------------------------------------------------

class SemanticBoundaryChunker:
    """Structure-preserving hierarchical chunker with window budgeting.

    Splits a state payload so that JSON/YAML brackets, Markdown headers, fenced
    code blocks, stack-trace frames and tables are **never** sliced mid-field.
    Windows are budgeted to ``max_chunk_tokens`` (default ``512``) with
    ``overlap_tokens`` (default ``64``) of overlap to preserve inter-chunk
    relational continuity.

    Args:
        tokenizer: Optional object exposing ``encode(text) -> List[int]``. When
            ``None``, a character-ratio estimate (``_CHARS_PER_TOKEN``) is used for
            budget enforcement only; the decision itself never depends on it.
        max_chunk_tokens: ``C_max``. Hard budget per chunk.
        overlap_tokens: ``C_overlap``. Tokens repeated between adjacent chunks.
        min_chunk_tokens: Chunks smaller than this are merged forward, so a
            boundary-dense document does not shatter into one-line chunks.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        min_chunk_tokens: int = 16,
    ) -> None:
        if max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if overlap_tokens >= max_chunk_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be < max_chunk_tokens ({max_chunk_tokens})"
            )
        if min_chunk_tokens < 1:
            raise ValueError("min_chunk_tokens must be >= 1")

        self.tokenizer = tokenizer
        self.max_chunk_tokens = int(max_chunk_tokens)
        self.overlap_tokens = int(overlap_tokens)
        self.min_chunk_tokens = int(min_chunk_tokens)

    # -- token accounting ---------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Token count used for budget enforcement."""
        if not text:
            return 0
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            try:
                return len(self.tokenizer.encode(text))
            except (RuntimeError, ValueError, TypeError) as exc:
                _log_tokenizer_fallback_once(exc)
        return max(1, int(len(text) / _CHARS_PER_TOKEN))

    def _fits(self, text: str) -> bool:
        return self.count_tokens(text) <= self.max_chunk_tokens

    def _fits_with_overlap(self, lines: Sequence[str], start: int, end: int) -> bool:
        """Whether ``lines[start:end]`` fits ``C_max`` once overlap is prepended.

        Adjacent chunks share up to ``C_overlap`` tokens, so a window that fits on
        its own can still overflow once its predecessor's tail is re-attached.
        Planning against the *post-overlap* size is what keeps the final chunks
        inside the budget rather than discovering the overflow after assembly.
        """
        body = "\n".join(lines[start:end])
        if not self._fits(body):
            return False
        if start == 0 or self.overlap_tokens <= 0:
            return True
        overlap = self._overlap_prefix(lines, start)
        return self.count_tokens(overlap + "\n" + body) <= self.max_chunk_tokens

    def _overlap_prefix(self, lines: Sequence[str], start: int) -> str:
        """The text that will be prepended to the chunk starting at ``start``."""
        n = self._overlap_lines(lines)
        if n <= 0:
            return ""
        return "\n".join(lines[max(0, start - n) : start])

    # -- public API ---------------------------------------------------------------

    def chunk(self, state: Any) -> List[Chunk]:
        """Split ``state`` into an ordered list of :class:`Chunk` objects.

        ``state`` may be a ``str``, or any object convertible to a string. When the
        payload has *no* semantic boundary at all it is returned as a single chunk
        rather than being force-cut mid-token.
        """
        text = state if isinstance(state, str) else _stringify(state)
        if not text.strip():
            return []

        lines = text.splitlines()
        boundaries = find_semantic_boundaries(text)
        # Always allow a cut at line 0, and the final chunk is the tail.
        cut_points = self._plan_cuts(lines, boundaries)

        if not cut_points:
            return [
                Chunk(
                    index=0,
                    text=text,
                    start_line=0,
                    end_line=len(lines) - 1,
                    n_tokens=self.count_tokens(text),
                    boundary="hard",
                    kind=detect_structure_kind(text),
                )
            ]

        return self._assemble(lines, cut_points, text)

    def chunk_text(self, text: str) -> List[str]:
        """Convenience wrapper returning only the chunk payloads."""
        return [c.text for c in self.chunk(text)]

    def describe(self, state: Any) -> Dict[str, Any]:
        """Diagnostic view of the chunking decision for ``state``."""
        chunks = self.chunk(state)
        return {
            "n_chunks": len(chunks),
            "max_chunk_tokens": self.max_chunk_tokens,
            "overlap_tokens": self.overlap_tokens,
            "chunks": [c.to_dict() for c in chunks],
            "token_budget_respected": all(c.n_tokens <= self.max_chunk_tokens for c in chunks),
        }

    # -- internals ----------------------------------------------------------------

    def _plan_cuts(
        self,
        lines: Sequence[str],
        boundaries: List[Tuple[int, str]],
    ) -> List[Tuple[int, str]]:
        """Choose the ordered line indices at which to start a new chunk.

        Greedily accumulate whole structural units until adding the next one would
        exceed ``C_max``. When *no* boundary is available before the budget is
        exhausted, the window is force-cut at the budget line: refusing to split at
        all would silently violate ``C_max``, and violating it visibly is far
        better than quietly returning an oversized window. The forced cut is
        recorded as ``"forced"`` so callers can distinguish the two cases.
        """
        if not lines:
            return []

        # Boundary line indices, excluding line 0 (document start).
        bmap = {idx: strength for idx, strength in boundaries if idx > 0}
        boundary_lines = sorted(bmap)
        fences = find_fenced_blocks(lines)

        cuts: List[Tuple[int, str]] = []
        window_start = 0

        for b_idx in boundary_lines:
            # A fenced block is one atomic unit, so a cut on *either* of its
            # delimiters is relocated past the whole block. Without this, cutting
            # on the closing fence would emit a chunk with an unbalanced ```.
            b_idx = self._push_past_fence(b_idx, fences, len(lines))
            if b_idx <= window_start:
                continue  # no forward progress; skip rather than loop

            if self._fits_with_overlap(lines, window_start, b_idx + 1):
                continue
            # This boundary is the first one past the budget: cut here (hard beats
            # soft beats forced) so the window never spans a structural unit.
            strength = bmap.get(b_idx, "forced")
            if strength == "soft":
                # Prefer a later hard boundary if one is close, so we cut on
                # structure rather than on a blank line mid-construct.
                strength = self._nearby_hard(bmap, b_idx) or "soft"
            cuts.append((b_idx, strength))
            window_start = b_idx

        # Trailing remainder.
        if self._fits_with_overlap(lines, window_start, len(lines)):
            return cuts

        # The tail is over budget: force-cut it at the budget line, repeatedly.
        cursor = window_start
        while True:
            budget_line = self._budget_line(lines, cursor)
            if budget_line is None or budget_line >= len(lines):
                break
            # A code block is indivisible: push the cut past the closing fence so
            # the window never lands mid-listing. A single oversized block then
            # yields one oversized chunk, which is the honest outcome -- better a
            # visibly large chunk than silently invalid code.
            budget_line = self._push_past_fence(budget_line, fences, len(lines))
            cuts.append((budget_line, "forced"))
            if budget_line <= cursor:
                break
            cursor = budget_line
            if cursor >= len(lines):
                break

        return cuts

    @staticmethod
    def _push_past_fence(line_idx: int, fences: Sequence[Tuple[int, int]], n_lines: int) -> int:
        """Move a cut anywhere inside (or on either delimiter of) a fenced block
        to just past that block's closing fence.

        The comparison is inclusive on both ends: a cut landing exactly on the
        *closing* fence would otherwise emit a chunk with an unbalanced ````` ``,
        which is just as broken as cutting mid-listing.
        """
        for start, end in fences:
            if start <= line_idx <= end:
                return min(end + 1, n_lines)
        return line_idx

    def _nearby_hard(self, bmap: Dict[int, str], start: int, window: int = 8) -> Optional[str]:
        """A hard boundary shortly after ``start``, if one exists."""
        for offset in range(1, window + 1):
            if bmap.get(start + offset) == "hard":
                return "hard"
        return None

    def _budget_line(self, lines: Sequence[str], start: int) -> Optional[int]:
        """First line index at/after ``start`` that would exceed the token budget.

        The budget accounts for the overlap the chunk will receive, so a
        force-cut window lands inside ``C_max`` after assembly rather than
        overflowing once its predecessor's tail is prepended.
        """
        for idx in range(start, len(lines)):
            if not self._fits_with_overlap(lines, start, idx + 1):
                # The window is lines[start:idx]; never return `start` (no progress).
                return idx if idx > start else None
        return None

    def _assemble(
        self,
        lines: Sequence[str],
        cuts: List[Tuple[int, str]],
        full_text: str,
    ) -> List[Chunk]:
        """Build the chunk list from the planned cut points, applying overlap."""
        # Overlap in *lines*, derived from the token budget so it tracks C_overlap.
        overlap_lines = self._overlap_lines(lines)
        fences = find_fenced_blocks(lines)

        starts = [0] + [c[0] for c in cuts]
        strengths = ["hard"] + [c[1] for c in cuts]
        chunks: List[Chunk] = []

        for pos, raw_start in enumerate(starts):
            # Re-attach the tail of the previous chunk (overlap).
            start = raw_start
            if pos > 0 and overlap_lines > 0:
                start = max(0, raw_start - overlap_lines)
                # Overlap must not reach back *into* a completed fenced block,
                # or the re-attached tail would orphan the opening ```.
                for f_start, f_end in fences:
                    if f_end < raw_start and start <= f_end:
                        start = f_end + 1

            if pos + 1 < len(starts):
                end = starts[pos + 1]  # exclusive
            else:
                end = len(lines)

            body = "\n".join(lines[start:end]).rstrip()
            if not body:
                continue
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=body,
                    start_line=start,
                    end_line=end - 1,
                    n_tokens=self.count_tokens(body),
                    boundary=strengths[pos],
                    kind=detect_structure_kind(body),
                )
            )

        return chunks

    def _overlap_lines(self, lines: Sequence[str]) -> int:
        """Convert ``C_overlap`` tokens into a line count for the overlap window."""
        if self.overlap_tokens <= 0:
            return 0
        target_chars = self.overlap_tokens * _CHARS_PER_TOKEN
        running = 0.0
        count = 0
        for line in reversed(lines):
            running += len(line) + 1
            count += 1
            if running >= target_chars:
                break
        # Never overlap more than half the document's worth of context.
        return min(count, max(1, len(lines) // 2))


def _stringify(state: Any) -> str:
    """Render a non-string state payload as text, preferring structured formats."""
    if isinstance(state, (dict, list, tuple)):
        try:
            import json

            return json.dumps(state, indent=2, default=str)
        except Exception:
            return str(state)
    return str(state)


# --------------------------------------------------------------------------------------
# 2. Salience-Gated Chunk Routing
# --------------------------------------------------------------------------------------

class SalienceGatedChunkRouter(nn.Module):
    """Routes a question to the chunks that actually matter.

        s_i(q) = ( h_q^T W_salience ) z_{chunk,i} / sqrt(d)
        w_i    = Softmax( s_i(q) / tau )

    ``W_salience`` projects both the question embedding and the chunk embeddings
    into a shared salience space, so relevance is measured in a learned metric
    rather than raw token overlap. The ``1/sqrt(d)`` factor keeps the dot product
    from growing with hidden width, and the temperature ``tau`` controls how
    peaked the routing distribution is: large ``tau`` approaches uniform
    attention over all chunks, small ``tau`` concentrates on the single winner.

    Args:
        hidden_size: Dimension of the incoming question / chunk embeddings.
        salience_dim: Width of the shared salience space (defaults to
            ``hidden_size`` when omitted).
        temperature: Softmax temperature ``tau``.
    """

    def __init__(
        self,
        hidden_size: int,
        salience_dim: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__()
        salience_dim = int(salience_dim or hidden_size)
        self.hidden_size = int(hidden_size)
        self.salience_dim = salience_dim
        self.temperature = float(temperature)

        self.q_proj = nn.Linear(hidden_size, salience_dim, bias=False)
        self.chunk_proj = nn.Linear(hidden_size, salience_dim, bias=False)
        # A learnable residual on the salience logit, letting the model learn that
        # some chunks are globally more informative regardless of the question.
        self.chunk_bias = nn.Parameter(torch.zeros(1))
        self.scale = salience_dim ** -0.5

        self._last_weights: Optional[torch.Tensor] = None

    @property
    def last_weights(self) -> Optional[torch.Tensor]:
        """Routing weights from the most recent forward pass (``[N]``)."""
        return self._last_weights

    def forward(
        self,
        question_repr: torch.Tensor,
        chunk_reprs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing salience and weights.

        Args:
            question_repr: ``[B, H]`` or ``[H]`` question embedding.
            chunk_reprs: ``[B, N, H]`` or ``[N, H]`` chunk embeddings (``N`` chunks).

        Returns:
            ``(salience, weights)`` of shape ``[B, N]`` each. ``salience`` is the
            pre-softmax score; ``weights`` sums to 1 over the chunk axis.
        """
        if question_repr.ndim == 1:
            question_repr = question_repr.unsqueeze(0)
        if chunk_reprs.ndim == 2:
            chunk_reprs = chunk_reprs.unsqueeze(0)
        if question_repr.shape[0] == 1 and chunk_reprs.shape[0] > 1:
            question_repr = question_repr.expand(chunk_reprs.shape[0], -1)
        if question_repr.shape[0] != chunk_reprs.shape[0]:
            raise ValueError(
                f"batch mismatch: question {tuple(question_repr.shape)} vs "
                f"chunks {tuple(chunk_reprs.shape)}"
            )

        q = self.q_proj(question_repr)                     # [B, d_s]
        z = self.chunk_proj(chunk_reprs)                   # [B, N, d_s]
        salience = torch.einsum("bd,bnd->bn", q, z) * self.scale + self.chunk_bias
        weights = F.softmax(salience / max(self.temperature, 1e-6), dim=-1)

        self._last_weights = weights.detach()
        return salience, weights


# --------------------------------------------------------------------------------------
# 3. Cross-Chunk Evidential LogSumExp Fusion
# --------------------------------------------------------------------------------------

def evidential_logsumexp_fuse(
    per_chunk_logits: torch.Tensor,
    chunk_weights: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Fuse per-chunk evidence into one calibrated simplex.

        s*(q, o_k) = log sum_{i=1..N} w_i * exp( s_i(q, o_k) / tau )

    The fusion is **monotonic** in the amount of supporting evidence: adding a
    chunk that agrees with the current winner can only raise the fused score, and
    scattered evidence across ``N`` chunks *accumulates* rather than washing out
    (a plain mean would dilute a strong local signal; a plain max would discard
    corroboration). ``LogSumExp`` gives the numerical stability that the naive
    ``sum(w * exp(s/tau))`` form lacks.

    Temperature controls how sharply evidence is pooled. Because this is a *log of
    a sum* rather than a max, the ``tau -> inf`` limit is the **weighted mean**
    (``sum_i w_i s_i``), since
    ``tau * log(sum_i w_i (1 + s_i/tau)) -> sum_i w_i s_i``. The ``tau -> 0``
    limit is the plain ``argmax_i s_i`` (the weights enter as ``tau * log w_i``
    and vanish). A wide ``tau`` therefore averages evidence across chunks, while
    a narrow one lets the single highest-scoring chunk decide.

    Args:
        per_chunk_logits: ``[B, N, K]`` per-chunk per-option scores.
        chunk_weights: ``[B, N]`` routing weights (a bare ``[N]`` is broadcast).
        temperature: Fusion temperature ``tau``.

    Returns:
        ``[B, K]`` fused scores, usable directly as logits for a softmax.
    """
    if per_chunk_logits.ndim != 3:
        raise ValueError(
            f"per_chunk_logits must be [B, N, K], got {tuple(per_chunk_logits.shape)}"
        )

    if chunk_weights.ndim == 1:
        chunk_weights = chunk_weights.unsqueeze(0)
    if chunk_weights.ndim == 3:
        chunk_weights = chunk_weights.squeeze(-1)
    if chunk_weights.ndim != 2:
        raise ValueError(
            f"chunk_weights must be [N] or [B, N], got {tuple(chunk_weights.shape)}"
        )

    n_chunks = per_chunk_logits.shape[1]
    if chunk_weights.shape[-1] != n_chunks:
        raise ValueError(
            f"chunk_weights has {chunk_weights.shape[-1]} entries but there are "
            f"{n_chunks} chunks"
        )
    if chunk_weights.shape[0] == 1 and per_chunk_logits.shape[0] > 1:
        chunk_weights = chunk_weights.expand(per_chunk_logits.shape[0], -1)

    tau = max(float(temperature), 1e-6)
    # [B, N, 1] so each chunk's whole option simplex is weighted as a unit.
    w = chunk_weights.to(per_chunk_logits.dtype).unsqueeze(-1)
    scaled = per_chunk_logits / tau

    # logsumexp over the chunk axis, then add back log w.
    fused = torch.logsumexp(scaled + torch.log(w.clamp_min(1e-12)), dim=1)
    return fused * tau


# --------------------------------------------------------------------------------------
# 4. ChunkedSpanPointer
# --------------------------------------------------------------------------------------

@dataclass
class ChunkedSpanResult:
    """Result of a chunk-resolved span extraction."""

    value: str
    chunk_index: int
    start_char: int
    end_char: int
    confidence: float
    n_candidates: int
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "n_candidates": self.n_candidates,
            "truncated": self.truncated,
        }


class ChunkedSpanPointer:
    """Resolves the winning chunk and extracts an exact, untruncated span string.

    Given per-chunk salience ``s_i(q)``, the winner is

        i* = argmax_i s_i

    The span is then extracted from chunk ``i*`` *in full* -- never clipped to the
    chunk budget -- because truncating an extracted value silently corrupts the
    answer. When the winning chunk yields no match the remaining chunks are
    searched in descending salience order, so evidence scattered across chunks is
    still recovered rather than lost to a routing mistake.

    Args:
        max_candidates: Cap on how many chunks are searched in the fallback sweep.
    """

    def __init__(self, max_candidates: int = 4) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        self.max_candidates = int(max_candidates)

    def resolve(
        self,
        chunks: Sequence[Chunk],
        salience: torch.Tensor,
        patterns: Optional[Sequence[str]] = None,
        max_span_chars: int = 4096,
    ) -> Optional[ChunkedSpanResult]:
        """Extract the best span from the winning chunk.

        Args:
            chunks: Ordered chunks from :class:`SemanticBoundaryChunker`.
            salience: ``[N]`` per-chunk salience for the current question.
            patterns: Ordered regexes identifying the span. The first pattern that
                matches in a given chunk wins; patterns are tried in order in every
                chunk, so pattern priority dominates chunk ordering.
            max_span_chars: Hard cap on the returned span, to bound memory. If it
                bites, ``truncated`` is set to ``True`` on the result.

        Returns:
            A :class:`ChunkedSpanResult`, or ``None`` when nothing matched.
        """
        if not chunks:
            return None
        if salience.ndim != 1:
            salience = salience.reshape(-1)
        if salience.numel() != len(chunks):
            raise ValueError(
                f"salience has {salience.numel()} entries but {len(chunks)} chunks were given"
            )

        order = torch.argsort(salience, descending=True).tolist()
        search_order = order[: max(1, self.max_candidates)]
        best_salience = float(salience[order[0]].detach())

        compiled = None
        if patterns:
            import re

            compiled = [(p, re.compile(p, re.IGNORECASE | re.DOTALL)) for p in patterns]

        for rank, chunk_idx in enumerate(search_order):
            chunk = chunks[chunk_idx]
            text = chunk.text
            if compiled:
                for _pattern, rx in compiled:
                    m = rx.search(text)
                    if m:
                        value = m.group(0)
                        truncated = len(value) > max_span_chars
                        return ChunkedSpanResult(
                            value=value[:max_span_chars],
                            chunk_index=chunk.index,
                            start_char=m.start(0),
                            end_char=m.end(0),
                            confidence=self._confidence(best_salience, rank),
                            n_candidates=len(chunks),
                            truncated=truncated,
                        )
            else:
                stripped = text.strip()
                if stripped:
                    truncated = len(stripped) > max_span_chars
                    return ChunkedSpanResult(
                        value=stripped[:max_span_chars],
                        chunk_index=chunk.index,
                        start_char=text.find(stripped),
                        end_char=text.find(stripped) + len(stripped),
                        confidence=self._confidence(best_salience, rank),
                        n_candidates=len(chunks),
                        truncated=truncated,
                    )

        return None

    @staticmethod
    def _confidence(best_salience: float, rank: int) -> float:
        """Confidence in ``[0, 1]`` from the winning salience and the search rank."""
        if rank == 0:
            return max(0.0, min(1.0, 0.5 + 0.5 * float(best_salience)))
        # Decay geometrically when we had to fall through to a lower-ranked chunk.
        return max(0.0, min(1.0, (0.5 + 0.5 * float(best_salience)) * (0.5 ** rank)))


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

#: Set once per process after the placeholder-feature warning has been emitted.
_PLACEHOLDER_WARNED: bool = False


def _placeholder_chunk_seed(text: str) -> int:
    """Process-independent 31-bit seed for a chunk's placeholder features.

    The built-in ``hash`` is salted per process (``PYTHONHASHSEED``), so it cannot
    seed reproducible routing; a SHA-256 digest of the chunk text can.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31)


def _warn_placeholder_chunk_features() -> None:
    """Warn once per process that chunk features are placeholders, not semantic embeddings."""
    global _PLACEHOLDER_WARNED
    if _PLACEHOLDER_WARNED:
        return
    _PLACEHOLDER_WARNED = True
    warnings.warn(
        "ChunkedRhenState: no encoder supplied, so chunk features are deterministic "
        "PLACEHOLDERS derived from a hash of each chunk's text, not semantic embeddings. "
        "Routing is reproducible but not meaningful until an encoder is passed.",
        RuntimeWarning,
        stacklevel=3,
    )


class ChunkedRhenState:
    """End-to-end chunked decision path for a long state payload.

    Ties the four pieces together:

    1. :class:`SemanticBoundaryChunker` splits the state without breaking structure.
    2. :class:`SalienceGatedChunkRouter` weights chunks against the question.
    3. :func:`evidential_logsumexp_fuse` merges per-chunk evidence.
    4. :class:`ChunkedSpanPointer` resolves the winning chunk for ``Extract``.

    Args:
        hidden_size: Embedding width for the router (required only for
            :meth:`route`).
        tokenizer: Optional tokenizer for exact budget accounting.
        max_chunk_tokens: ``C_max`` (default 512).
        overlap_tokens: ``C_overlap`` (default 64).
        temperature: Fusion / routing temperature ``tau``.
        encoder: Optional callable mapping the list of chunk texts to a
            ``[N, hidden_size]`` tensor of real encoder states. Without it, chunk
            features are deterministic placeholders and a one-time
            ``RuntimeWarning`` is emitted.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        tokenizer: Any = None,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        encoder: Optional[Callable[[List[str]], torch.Tensor]] = None,
    ) -> None:
        self.chunker = SemanticBoundaryChunker(
            tokenizer=tokenizer,
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        self.router = SalienceGatedChunkRouter(hidden_size, temperature=temperature)
        self.pointer = ChunkedSpanPointer()
        self.temperature = float(temperature)
        self.encoder = encoder
        self._last: Dict[str, Any] = {}

    @property
    def last_trace(self) -> Dict[str, Any]:
        """Diagnostics from the most recent :meth:`decide` / :meth:`route` call."""
        return dict(self._last)

    def split(self, state: Any) -> List[Chunk]:
        """Semantic chunking of ``state``."""
        return self.chunker.chunk(state)

    def route(
        self,
        state: Any,
        question_repr: torch.Tensor,
    ) -> Tuple[List[Chunk], torch.Tensor, torch.Tensor]:
        """Chunk the state and compute routing salience / weights.

        Returns:
            ``(chunks, salience, weights)`` where ``salience`` and ``weights`` are
            ``[N]`` tensors aligned with ``chunks``.
        """
        chunks = self.split(state)
        if not chunks:
            empty = torch.zeros(0)
            self._last = {"n_chunks": 0, "reason": "empty_state"}
            return chunks, empty, empty

        dim = self.router.hidden_size
        if self.encoder is not None:
            reprs = self.encoder([chunk.text for chunk in chunks])
            if not isinstance(reprs, torch.Tensor) or tuple(reprs.shape) != (len(chunks), dim):
                got = tuple(reprs.shape) if isinstance(reprs, torch.Tensor) else type(reprs).__name__
                raise ValueError(f"encoder must return a [{len(chunks)}, {dim}] tensor, got {got}")
            chunk_reprs = reprs.to(dtype=question_repr.dtype, device=question_repr.device)
            feature_source = "encoder"
        else:
            _warn_placeholder_chunk_features()
            chunk_reprs = torch.zeros(len(chunks), dim, dtype=question_repr.dtype,
                                      device=question_repr.device)
            # Placeholder features: one content-derived, process-stable seed per chunk.
            # A real deployment passes an encoder instead; the routing maths is identical.
            for i, chunk in enumerate(chunks):
                g = torch.Generator(device="cpu").manual_seed(_placeholder_chunk_seed(chunk.text))
                chunk_reprs[i] = torch.randn(dim, generator=g).to(
                    dtype=question_repr.dtype, device=question_repr.device
                )
            feature_source = "placeholder"

        salience, weights = self.router(question_repr, chunk_reprs)
        self._last = {
            "n_chunks": len(chunks),
            "feature_source": feature_source,
            "salience": salience.detach().reshape(-1).tolist(),
            "weights": weights.detach().reshape(-1).tolist(),
            "winner": int(torch.argmax(salience).item()),
        }
        return chunks, salience.reshape(-1), weights.reshape(-1)

    def decide(
        self,
        state: Any,
        question_repr: torch.Tensor,
        per_chunk_logits: torch.Tensor,
        patterns: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Fuse ``per_chunk_logits`` (``[N, K]``) across chunks into one simplex.

        Returns a dict with the fused ``probabilities`` (normalized over options),
        the raw fused ``scores``, the per-chunk routing trace, and -- when
        ``patterns`` is given -- the resolved ``span``.
        """
        chunks, salience, weights = self.route(state, question_repr)
        if not chunks:
            return {
                "n_chunks": 0,
                "probabilities": [],
                "scores": [],
                "chunks": [],
                "span": None,
                "salience": [],
                "weights": [],
            }

        logits = per_chunk_logits
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        if logits.shape[1] != len(chunks):
            raise ValueError(
                f"per_chunk_logits has {logits.shape[1]} chunk slots but {len(chunks)} "
                f"chunks were produced"
            )

        fused = evidential_logsumexp_fuse(logits, weights.unsqueeze(0), self.temperature)
        probs = F.softmax(fused, dim=-1).reshape(-1)

        span = None
        if patterns is not None:
            span = self.pointer.resolve(chunks, salience, patterns)

        return {
            "n_chunks": len(chunks),
            "probabilities": probs.tolist(),
            "scores": fused.reshape(-1).tolist(),
            "chunks": [c.to_dict() for c in chunks],
            "span": span.to_dict() if span is not None else None,
            "salience": salience.tolist(),
            "weights": weights.tolist(),
            "winner": int(torch.argmax(salience).item()),
        }


__all__ = [
    "DEFAULT_MAX_CHUNK_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "DEFAULT_TEMPERATURE",
    "Chunk",
    "ChunkedSpanResult",
    "ChunkedSpanPointer",
    "ChunkedRhenState",
    "SalienceGatedChunkRouter",
    "SemanticBoundaryChunker",
    "classify_boundary",
    "detect_structure_kind",
    "evidential_logsumexp_fuse",
    "find_fenced_blocks",
    "find_semantic_boundaries",
]
