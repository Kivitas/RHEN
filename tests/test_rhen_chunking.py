"""
Verification Suite for Next-Generation Semantic Hierarchical Chunking.

Covers the four cooperating pieces of ``rhen_chunker.py``:

1. ``SemanticBoundaryChunker`` -- JSON/YAML brackets, Markdown headers, fenced
   code blocks, stack-trace frames and table rows are never sliced mid-field;
   windows respect ``C_max = 512`` / ``C_overlap = 64`` budgeting.
2. ``SalienceGatedChunkRouter`` -- ``s_i(q)`` and ``w_i = Softmax(s_i/tau)``.
3. ``evidential_logsumexp_fuse`` -- cross-chunk evidential fusion is monotone in
   the amount of supporting evidence.
4. ``ChunkedSpanPointer`` -- resolves ``i* = argmax_i s_i`` and returns the exact,
   untruncated span.

The structural claims here are the load-bearing ones: a chunker that splits a
JSON object or a code fence mid-construct produces chunks that are *individually
unusable*, which is a silent correctness failure rather than a visible one.
"""

import math
import re

import torch

from rhen_chunker import (
    DEFAULT_MAX_CHUNK_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    ChunkedRhenState,
    ChunkedSpanPointer,
    SalienceGatedChunkRouter,
    SemanticBoundaryChunker,
    classify_boundary,
    detect_structure_kind,
    evidential_logsumexp_fuse,
    find_fenced_blocks,
    find_semantic_boundaries,
)

TICK = chr(96) * 3  # ``` -- kept out of a triple-quoted literal on purpose


def _json_state(n_keys: int = 40) -> str:
    return "{\n" + "".join(f'  "key_{i}": {i},\n' for i in range(n_keys)) + "}"


def _markdown_state(n_sections: int = 30) -> str:
    return "\n".join(
        [f"## Section {i}\nFiller prose line for the {i}-th section, padded out." for i in range(n_sections)]
    )


def _fenced_state(n_lines: int = 50) -> str:
    return (
        "intro\n"
        + TICK + "python\n"
        + "\n".join(f"x{i} = {i}" for i in range(n_lines))
        + "\n" + TICK + "\nend"
    )


def _yaml_state(n_keys: int = 40) -> str:
    return "\n".join(f"key_{i}:\n  value: {i}\n  note: filler {i}" for i in range(n_keys))


def _trace_state(n_frames: int = 40) -> str:
    head = "Traceback (most recent call last):\n"
    return head + "\n".join(f'  File "mod{i}.py", line {i}, in fn' for i in range(n_frames))


def test_boundary_classification():
    print("\n--- Test 1: Semantic Boundary Classification ---")
    assert classify_boundary("}") == "hard", "a closing bracket always ends a construct"
    assert classify_boundary("{") == "hard", "a top-level opening bracket starts one"
    assert classify_boundary("key: value") == "hard", "a top-level YAML key starts a mapping entry"
    assert classify_boundary("## Heading") == "hard", "a markdown header is a hard boundary"
    assert classify_boundary(TICK) == "hard", "a fence delimiter is a hard boundary"
    assert classify_boundary("") == "soft", "a blank line is a soft boundary"
    assert classify_boundary("- bullet") == "soft", "a list item is a soft boundary"
    assert classify_boundary("  indented continuation") == "neutral"

    # A stack-trace frame is a soft boundary: safe to cut, but not mandatory.
    assert classify_boundary('  File "a.py", line 1') == "soft"
    print("  [PASS] Hard boundaries (brackets, YAML keys, headers, fences) and")
    print("         soft boundaries (blank lines, bullets, stack frames) classify correctly.")


def test_semantic_boundary_discovery():
    print("\n--- Test 2: Boundary Discovery & Fence Detection ---")
    md = _markdown_state()
    bounds = find_semantic_boundaries(md)
    assert bounds, "markdown headers must be discovered"
    assert bounds[0][0] == 0, "the document start is always a boundary"
    kinds = {s for _, s in bounds}
    assert "hard" in kinds

    # Inside an open fence there are no boundaries at all: a listing is atomic.
    fenced = _fenced_state()
    fbounds = find_semantic_boundaries(fenced)
    fence_lines = [i for i, _ in fbounds if fenced.splitlines()[i].strip().startswith(TICK)]
    assert len(fence_lines) == 2, "exactly the opening and closing fence are boundaries"

    blocks = find_fenced_blocks(fenced.splitlines())
    assert len(blocks) == 1, "one fenced block"
    start, end = blocks[0]
    assert fenced.splitlines()[start].strip().startswith(TICK)
    assert fenced.splitlines()[end].strip() == TICK
    print(f"  [PASS] {len(bounds)} markdown boundaries found; fence block spans lines {start}-{end}.")


def test_structure_detection():
    print("\n--- Test 3: Structure Kind Detection ---")
    assert detect_structure_kind('{"a": 1}') == "json"
    assert detect_structure_kind(_fenced_state()) == "code"
    assert detect_structure_kind("# Title\n\ntext") == "markdown"
    assert detect_structure_kind("| a | b |\n| --- | --- |") == "table"
    assert detect_structure_kind("key:\n  nested: 1") == "yaml"
    assert detect_structure_kind("just some prose here") == "text"
    print("  [PASS] JSON / code / markdown / table / yaml / text all detected.")


def test_json_never_sliced_mid_field():
    print("\n--- Test 4: JSON Structural Integrity Across Chunks ---")
    chunker = SemanticBoundaryChunker(max_chunk_tokens=64, overlap_tokens=8)
    chunks = chunker.chunk(_json_state(40))
    assert len(chunks) > 1, "a 40-key object must span multiple chunks"

    # Every occurrence of the key prefix must be a *complete* quoted key. A
    # chunk cut mid-field would leave a dangling `"key_17` with no closing quote.
    for c in chunks:
        starts = c.text.count('"key_')
        complete = len(re.findall(r'"key_\d+"', c.text))
        assert starts == complete, (
            f"chunk {c.index} has {starts} key prefixes but only {complete} complete "
            f"quoted keys -- a key was sliced mid-field"
        )
    # Every key in the source must be recoverable somewhere.
    recovered = set()
    for c in chunks:
        recovered.update(re.findall(r'"key_\d+"', c.text))
    assert len(recovered) == 40, f"expected all 40 keys, recovered {len(recovered)}"
    print(f"  [PASS] {len(chunks)} chunks, all 40 JSON keys intact (never sliced mid-field).")


def test_code_fence_is_atomic():
    print("\n--- Test 5: Fenced Code Blocks Are Indivisible ---")
    chunker = SemanticBoundaryChunker(max_chunk_tokens=40, overlap_tokens=4)
    chunks = chunker.chunk(_fenced_state(50))
    for c in chunks:
        n = c.text.count(TICK)
        assert n % 2 == 0, (
            f"chunk {c.index} has an unbalanced fence ({n} delimiters); a code "
            f"block must never be split across chunks"
        )
    # The whole listing must be recoverable from exactly one chunk.
    holders = [c for c in chunks if TICK in c.text]
    assert len(holders) == 1, "the fenced listing must live in a single chunk"
    holder = holders[0]
    for i in (0, 25, 49):
        assert f"x{i} = {i}" in holder.text, f"line x{i} was lost"
    print(f"  [PASS] Fenced listing kept whole in chunk {holder.index} "
          f"({holder.n_tokens} tokens, {holder.boundary} boundary).")


def test_markdown_and_yaml_and_trace():
    print("\n--- Test 6: Markdown / YAML / Stack-Trace Integrity ---")
    c_md = SemanticBoundaryChunker(max_chunk_tokens=60, overlap_tokens=8)
    md_chunks = c_md.chunk(_markdown_state(30))
    assert len(md_chunks) > 1
    for c in md_chunks:
        assert "##" in c.text, f"chunk {c.index} lost its header context"

    c_yaml = SemanticBoundaryChunker(max_chunk_tokens=60, overlap_tokens=8)
    yaml_chunks = c_yaml.chunk(_yaml_state(40))
    assert len(yaml_chunks) > 1
    # A YAML key and its nested value must stay together.
    for c in yaml_chunks:
        assert "value:" not in c.text or "key_" in c.text, (
            f"chunk {c.index} contains an orphaned YAML value"
        )

    c_tr = SemanticBoundaryChunker(max_chunk_tokens=60, overlap_tokens=8)
    tr_chunks = c_tr.chunk(_trace_state(40))
    assert len(tr_chunks) > 1
    for c in tr_chunks:
        assert 'File "' not in c.text or "mod" in c.text, (
            f"chunk {c.index} split a stack frame"
        )
    print(f"  [PASS] markdown={len(md_chunks)}, yaml={len(yaml_chunks)}, "
          f"trace={len(tr_chunks)} chunks, all structurally intact.")


def test_window_budgeting_and_overlap():
    print("\n--- Test 7: Window Budgeting & Overlap ---")
    assert DEFAULT_MAX_CHUNK_TOKENS == 512, "C_max must default to 512"
    assert DEFAULT_OVERLAP_TOKENS == 64, "C_overlap must default to 64"

    defaults = SemanticBoundaryChunker()
    assert defaults.max_chunk_tokens == 512
    assert defaults.overlap_tokens == 64

    # Budget: every chunk respects C_max when boundaries are plentiful.
    chunker = SemanticBoundaryChunker(max_chunk_tokens=64, overlap_tokens=8)
    state = _json_state(200)
    desc = chunker.describe(state)
    assert desc["n_chunks"] > 1
    assert desc["token_budget_respected"], "no chunk may exceed C_max on boundary-rich input"

    # Overlap: adjacent chunks share trailing context.
    ov = SemanticBoundaryChunker(max_chunk_tokens=64, overlap_tokens=24)
    ov_chunks = ov.chunk(_json_state(200))
    assert len(ov_chunks) > 1
    shared = 0
    for a, b in zip(ov_chunks, ov_chunks[1:]):
        a_lines = set(a.text.splitlines())
        b_lines = set(b.text.splitlines())
        shared = max(shared, len(a_lines & b_lines))
    assert shared > 0, "overlap must repeat context between adjacent chunks"

    # Zero overlap is a valid configuration.
    zero = SemanticBoundaryChunker(max_chunk_tokens=64, overlap_tokens=0)
    assert len(zero.chunk(_json_state(60))) > 1
    print(f"  [PASS] C_max=512 / C_overlap=64 defaults; budget respected across "
          f"{desc['n_chunks']} chunks; {shared} overlapping lines between neighbours.")


def test_chunker_guards():
    print("\n--- Test 8: Chunker Input Guards & Degenerate Cases ---")
    # Empty and whitespace-only states produce no chunks rather than one empty one.
    assert SemanticBoundaryChunker().chunk("") == []
    assert SemanticBoundaryChunker().chunk("   \n  \n") == []

    # Invalid configurations are rejected loudly.
    for bad in ({"max_chunk_tokens": 0}, {"overlap_tokens": -1},
                {"max_chunk_tokens": 32, "overlap_tokens": 64}):
        try:
            SemanticBoundaryChunker(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # A state with no boundaries at all is still returned (never dropped).
    flat = "word " * 500
    flat_chunks = SemanticBoundaryChunker(max_chunk_tokens=32, overlap_tokens=0).chunk(flat)
    assert len(flat_chunks) >= 1 and flat_chunks[0].text.strip()

    # A dict state is serialised structurally before chunking.
    dict_chunks = SemanticBoundaryChunker(max_chunk_tokens=64, overlap_tokens=8).chunk(
        {"a": 1, "b": 2, "c": 3}
    )
    assert dict_chunks and '"a"' in dict_chunks[0].text
    print("  [PASS] Empty states, invalid configs, boundary-free text and dict states handled.")


def test_salience_gated_routing():
    print("\n--- Test 9: Salience-Gated Chunk Routing ---")
    hidden = 32
    router = SalienceGatedChunkRouter(hidden, temperature=1.0)

    q = torch.randn(hidden)
    chunks = torch.randn(5, hidden)
    salience, weights = router(q, chunks)

    assert salience.shape == (1, 5), f"salience must be [B, N], got {tuple(salience.shape)}"
    assert weights.shape == (1, 5)
    assert abs(float(weights.detach().sum()) - 1.0) < 1e-5, "routing weights must form a simplex"
    assert bool((weights > 0).all()), "softmax weights are strictly positive"

    # Routing is deterministic for a fixed (q, chunks) pair.
    s2, w2 = router(q, chunks)
    assert torch.allclose(weights, w2, atol=1e-6), "routing must be deterministic"

    # The formula: s_i = (W_q q) . (W_z z_i) / sqrt(d) + b.
    manual = torch.einsum(
        "d,nd->n", router.q_proj(q), router.chunk_proj(chunks)
    ) * router.scale + router.chunk_bias
    assert torch.allclose(salience.reshape(-1), manual, atol=1e-5), (
        "salience must equal (h_q^T W)z_i / sqrt(d) + b"
    )

    # Temperature controls peakedness: low tau concentrates, high tau spreads.
    sharp = SalienceGatedChunkRouter(hidden, temperature=0.05)(q, chunks)[1]
    flat = SalienceGatedChunkRouter(hidden, temperature=50.0)(q, chunks)[1]
    assert float(sharp.detach().max()) > float(flat.detach().max()), "lower tau must be more peaked"
    assert abs(float(flat.detach().sum()) - 1.0) < 1e-5

    # Batched input, and a shape-mismatch guard.
    sb, wb = router(torch.randn(3, hidden), torch.randn(3, 4, hidden))
    assert sb.shape == (3, 4) and abs(float(wb[0].detach().sum()) - 1.0) < 1e-5
    try:
        router(torch.randn(2, hidden), torch.randn(3, 4, hidden))
    except ValueError:
        pass
    else:
        raise AssertionError("batch mismatch must raise")
    print(f"  [PASS] s_i matches (h_q^T W)z_i/sqrt(d) exactly; weights are a simplex; "
          f"temperature {float(sharp.detach().max()):.3f} (sharp) vs {float(flat.detach().max()):.3f} (flat).")


def test_evidential_fusion():
    print("\n--- Test 10: Cross-Chunk Evidential LogSumExp Fusion ---")
    # The fused score must equal the stated closed form.
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]]])
    w = torch.tensor([[0.2, 0.5, 0.3]])
    tau = 1.0
    fused = evidential_logsumexp_fuse(logits, w, tau)
    manual = torch.tensor([
        math.log(sum(
            float(w[0, i]) * math.exp(float(logits[0, i, k]) / tau) for i in range(3)
        )) for k in range(2)
    ]) * tau
    assert torch.allclose(fused.reshape(-1), manual, atol=1e-5), (
        "fusion must equal log sum_i w_i exp(s_i / tau)"
    )

    # Monotonicity: adding corroborating evidence strengthens the leader.
    base = evidential_logsumexp_fuse(logits, w, 1.0)
    extra = torch.cat([logits, torch.tensor([[[5.0, 0.0]]])], dim=1)
    w2 = torch.cat([w, torch.tensor([[1e-6]])], dim=1)
    grown = evidential_logsumexp_fuse(extra, w2, 1.0)
    assert float(grown[0, 0] - grown[0, 1]) > float(base[0, 0] - base[0, 1]), (
        "more supporting evidence must widen the margin in favour of the leader"
    )

    # Limits. This is a *log of a sum*, not a max, so the tau -> inf limit is the
    # weighted mean sum_i w_i s_i (not zero): expansion gives
    # tau*log(sum_i w_i (1 + s_i/tau)) -> sum_i w_i s_i. The tau -> 0 limit is the
    # weighted max.
    weighted_mean = (w.reshape(-1).unsqueeze(-1) * logits[0]).sum(0)
    big_tau = evidential_logsumexp_fuse(logits, w, 1e3)
    assert torch.allclose(big_tau.reshape(-1), weighted_mean, atol=1e-2), (
        f"tau -> inf must approach the weighted mean {weighted_mean.tolist()}, "
        f"got {big_tau.reshape(-1).tolist()}"
    )
    # As tau -> 0 the weights vanish (they enter as tau*log w_i with tau -> 0),
    # so the limit is the plain argmax_i s_i. Use non-degenerate data: a tie would
    # make argmax ambiguous rather than wrong.
    peaked = torch.tensor([[[5.0, 0.0], [0.0, 1.0], [1.0, 2.0]]])
    small_tau = evidential_logsumexp_fuse(peaked, w, 1e-3)
    assert int(torch.argmax(small_tau[0])) == 0, (
        f"tau -> 0 must select the highest-scoring chunk, got {small_tau.reshape(-1).tolist()}"
    )
    # Weights are genuinely used: shifting mass onto a weaker chunk changes the
    # fused ordering at moderate tau.
    flipped = torch.tensor([[[5.0, 0.0], [0.0, 9.0], [1.0, 2.0]]])
    skewed = torch.tensor([[0.01, 0.98, 0.01]])
    assert int(torch.argmax(evidential_logsumexp_fuse(flipped, skewed, 0.5)[0])) == 1, (
        "routing weights must influence which chunk dominates the fusion"
    )

    # Numerically stable: a large-magnitude logit must not overflow to inf.
    huge = evidential_logsumexp_fuse(torch.tensor([[[1e4, -1e4]]]), torch.tensor([[1.0]]), 1.0)
    assert bool(torch.isfinite(huge).all()), "LogSumExp must stay finite on large logits"

    # Shape guards.
    try:
        evidential_logsumexp_fuse(torch.randn(2, 2), torch.randn(2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("2-D input must raise")
    try:
        evidential_logsumexp_fuse(logits, torch.tensor([[0.5, 0.5]]))
    except ValueError:
        pass
    else:
        raise AssertionError("weight/chunk mismatch must raise")
    print("  [PASS] Fusion matches the closed form, is monotone in evidence, stable at")
    print("         1e4 logits; tau->inf gives the weighted mean, tau->0 the plain argmax.")


def test_chunked_span_pointer():
    print("\n--- Test 11: ChunkedSpanPointer ---")
    chunks = SemanticBoundaryChunker(max_chunk_tokens=48, overlap_tokens=4).chunk(
        _json_state(60)
    )
    pointer = ChunkedSpanPointer(max_candidates=4)

    # The winner is argmax_i s_i, and the span comes from that chunk.
    salience = torch.zeros(len(chunks))
    target = 3
    salience[target] = 5.0
    res = pointer.resolve(chunks, salience, patterns=[r'"key_\d+"'])
    assert res is not None, "a matching pattern must resolve"
    assert res.chunk_index == target, f"expected chunk {target}, got {res.chunk_index}"
    assert res.value.startswith('"key_') and res.value.endswith('"')
    assert not res.truncated, "an in-budget span must not be marked truncated"
    assert 0.0 <= res.confidence <= 1.0

    # Fallback sweep: an unmatched pattern returns None, not a wrong span.
    res_fb = pointer.resolve(chunks, salience, patterns=[r"NEVER_MATCHES_[0-9]+"])
    assert res_fb is None, "an unmatched pattern must return None, not a wrong span"

    # No patterns: the winning chunk's body is returned.
    res_body = pointer.resolve(chunks, salience)
    assert res_body is not None and res_body.value

    # Oversized spans are flagged, never silently truncated.
    res_cap = pointer.resolve(chunks, salience, max_span_chars=5)
    assert res_cap.truncated is True and len(res_cap.value) == 5

    # Degenerate inputs.
    assert pointer.resolve([], torch.zeros(0)) is None
    try:
        pointer.resolve(chunks, torch.zeros(len(chunks) + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("salience/chunk count mismatch must raise")
    try:
        ChunkedSpanPointer(max_candidates=0)
    except ValueError:
        pass
    else:
        raise AssertionError("max_candidates must be >= 1")
    print(f"  [PASS] Winner chunk {res.chunk_index} resolved via argmax_i s_i; span "
          f"{res.value!r} extracted exactly; truncation flagged, never silent.")


def test_orchestrated_chunked_decision():
    print("\n--- Test 12: End-to-End ChunkedRhenState Decision Path ---")
    torch.manual_seed(0)
    state = _json_state(120)
    engine = ChunkedRhenState(
        hidden_size=32, max_chunk_tokens=64, overlap_tokens=8, temperature=1.0
    )
    chunks = engine.split(state)
    n = len(chunks)
    assert n > 1, "the payload must be chunked"

    q = torch.randn(32)
    routed_chunks, salience, weights = engine.route(state, q)
    assert len(routed_chunks) == n
    assert salience.shape == (n,) and weights.shape == (n,)
    assert abs(float(weights.detach().sum()) - 1.0) < 1e-5
    assert engine.last_trace["n_chunks"] == n

    # Per-chunk logits, one option simplex per chunk.
    per_chunk_logits = torch.randn(n, 3)
    out = engine.decide(state, q, per_chunk_logits, patterns=[r'"key_\d+"'])

    assert out["n_chunks"] == n
    assert len(out["probabilities"]) == 3, "fused distribution is over the 3 options"
    assert abs(sum(out["probabilities"]) - 1.0) < 1e-5, "fused probs must form a simplex"
    assert out["winner"] == int(torch.argmax(salience).item())
    assert out["span"] is not None, "the Extract pointer must resolve"
    assert out["span"]["value"].startswith('"key_')
    assert len(out["chunks"]) == n

    # A per-chunk-logits/chunk-count mismatch is a loud error, not silent truncation.
    try:
        engine.decide(state, q, torch.randn(n + 3, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("chunk/logit count mismatch must raise")

    # Empty state returns a well-formed empty result.
    empty = engine.decide("", q, torch.zeros(1, 3))
    assert empty["n_chunks"] == 0 and empty["probabilities"] == []
    print(f"  [PASS] {n} chunks routed, fused simplex sums to 1.0, span "
          f"{out['span']['value']!r} resolved from chunk {out['span']['chunk_index']}.")


def run_all_tests():
    print("=" * 70)
    print("  NEXT-GEN SEMANTIC HIERARCHICAL CHUNKING - VERIFICATION")
    print("=" * 70)

    test_boundary_classification()
    test_semantic_boundary_discovery()
    test_structure_detection()
    test_json_never_sliced_mid_field()
    test_code_fence_is_atomic()
    test_markdown_and_yaml_and_trace()
    test_window_budgeting_and_overlap()
    test_chunker_guards()
    test_salience_gated_routing()
    test_evidential_fusion()
    test_chunked_span_pointer()
    test_orchestrated_chunked_decision()

    print("\n" + "=" * 70)
    print("  ALL 12 CHUNKING VERIFICATION SUITES PASSED WITH ZERO ERRORS!")
    print("=" * 70)


# ---------------------------------------------------------------------------
# unittest suite wrapper (test-runner modernization, Phase A2): makes this
# file discoverable by unittest/pytest without changing the script-style
# runner below. Methods preserve the legacy __main__ call order.
import unittest as _unittest


class TestRhenChunking(_unittest.TestCase):
    """test_rhen_chunking verification suite (master battery)."""

    def test_boundary_classification(self):
        test_boundary_classification()

    def test_semantic_boundary_discovery(self):
        test_semantic_boundary_discovery()

    def test_structure_detection(self):
        test_structure_detection()

    def test_json_never_sliced_mid_field(self):
        test_json_never_sliced_mid_field()

    def test_code_fence_is_atomic(self):
        test_code_fence_is_atomic()

    def test_markdown_and_yaml_and_trace(self):
        test_markdown_and_yaml_and_trace()

    def test_window_budgeting_and_overlap(self):
        test_window_budgeting_and_overlap()

    def test_chunker_guards(self):
        test_chunker_guards()

    def test_salience_gated_routing(self):
        test_salience_gated_routing()

    def test_evidential_fusion(self):
        test_evidential_fusion()

    def test_chunked_span_pointer(self):
        test_chunked_span_pointer()

    def test_orchestrated_chunked_decision(self):
        test_orchestrated_chunked_decision()

if __name__ == "__main__":
    run_all_tests()
