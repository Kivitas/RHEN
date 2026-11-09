"""
Rhen Delimiter Packaging Engine.

Transforms arbitrary states (JSON, dicts, strings, Markdown) and multi-question
decision schemas into packed token layouts with exact coordinate tracking for
non-autoregressive latent pointer readout.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import torch

try:
    from rhen.tokenizer import (
        STATE_TOKEN,
        STATE_DIFF_TOKEN,
        QUESTION_TOKEN,
        OPTION_START_TOKEN,
        OPTION_END_TOKEN,
        DECIDE_TOKEN,
        MULTICHOICE_TOKEN,
        RANGE_TOKEN,
        EXTRACT_TOKEN,
        LATENT_SLOT_TOKEN,
        get_tokenizer,
    )
except ImportError:
    from tokenizer import (
        STATE_TOKEN,
        STATE_DIFF_TOKEN,
        QUESTION_TOKEN,
        OPTION_START_TOKEN,
        OPTION_END_TOKEN,
        DECIDE_TOKEN,
        MULTICHOICE_TOKEN,
        RANGE_TOKEN,
        EXTRACT_TOKEN,
        LATENT_SLOT_TOKEN,
        get_tokenizer,
    )

DELIMITER_REPLACEMENTS = {
    "<|state|>": "<¦state¦>",
    "<|state_diff|>": "<¦state_diff¦>",
    "<|q|>": "<¦q¦>",
    "<|opt|>": "<¦opt¦>",
    "<|/opt|>": "<¦/opt¦>",
    "<|decide|>": "<¦decide¦>",
    "<|multichoice|>": "<¦multichoice¦>",
    "<|range|>": "<¦range¦>",
    "<|extract|>": "<¦extract¦>",
    "<|latent_slot|>": "<¦latent_slot¦>",
}

def sanitize_delimiter_text(text: str, allow_latent_slots: bool = False) -> str:
    """
    Neutralizes control-token syntax in untrusted text (state, questions, options)
    so it cannot be read as a Rhen delimiter or as a model special token.

    1. The ten Rhen delimiters are replaced by their safe forms (DELIMITER_REPLACEMENTS).
    2. Every remaining ``<|`` becomes ``<¦`` and every ``|>`` becomes ``¦>``, which
       neutralizes any other control-token-looking sequence, such as ``<|system|>``,
       ``<|im_end|>`` or ``<|endoftext|>``.

    With ``allow_latent_slots=True`` the literal ``<|latent_slot|>`` token is kept intact
    for trusted latent grounding; the text around it is still neutralized.
    """
    if not isinstance(text, str):
        text = str(text)
    for token, safe in DELIMITER_REPLACEMENTS.items():
        if allow_latent_slots and token == LATENT_SLOT_TOKEN:
            continue
        if token in text:
            text = text.replace(token, safe)
    # Split around the allowed slot token so that it is never touched by the generic pass.
    segments = text.split(LATENT_SLOT_TOKEN) if allow_latent_slots else [text]
    neutralized = [seg.replace("<|", "<¦").replace("|>", "¦>") for seg in segments]
    return LATENT_SLOT_TOKEN.join(neutralized)


def compute_structural_depth_ids(text: str) -> List[int]:
    """
    Assigns an integer depth level (0 to 15) to each line/token based on
    JSON/YAML indentation (e.g. 2 spaces = 1 depth level, clamped to 15).
    """
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return []
    lines = text.splitlines()
    if not lines:
        return [0] if text else []
    depth_ids = []
    for line in lines:
        expanded = line.replace("\t", "  ")
        leading_spaces = len(expanded) - len(expanded.lstrip(" "))
        depth = min(15, max(0, leading_spaces // 2))
        depth_ids.append(depth)
    return depth_ids


@dataclass
class PackedBranchMeta:
    """Metadata for a single decision branch packed within a Rhen sequence."""
    question_text: str
    options: List[str]
    branch_token_len: int
    decide_idx: int  # Absolute sequence index of <|decide|> (query representation h_q)
    option_end_indices: List[int]  # Absolute sequence indices of each <|/opt|> (option representations h_o)
    num_options: int
    primitive_type: str = "choice"  # 'noul', 'choice', 'score', 'multichoice', 'range', 'extract'
    metadata: Optional[Dict[str, Any]] = None
    parent_branch_idx: Optional[int] = None  # For 2D speculative tree branches
    condition_option_idx: Optional[int] = None  # Which option of parent activates this branch
    # Absolute index of each option's <|opt|> token. With option_end_indices it delimits the
    # option's span, which the decision head pools over (None for hand-built metadata).
    option_start_indices: Optional[List[int]] = None


@dataclass
class PackedRhenSequence:
    """Full packed sequence ready for EOS LLM forward pass with Rhen pointer heads."""
    input_ids: torch.Tensor  # Shape: (1, seq_len)
    attention_mask: torch.Tensor  # 4D 2D-causal mask: (1, 1, seq_len, seq_len)
    state_length: int
    branch_lengths: List[int]
    branches: List[PackedBranchMeta]
    depth_ids: Optional[List[int]] = None
    branch_depth_ids: Optional[torch.Tensor] = None
    is_delta: bool = False
    latent_tensors: Optional[Dict[str, torch.Tensor]] = None
    latent_slot_indices: Optional[List[int]] = None
    is_speculative_tree: bool = False


def _parse_question_item(item: Any) -> Tuple[str, List[str], str, Optional[Dict[str, Any]]]:
    """Extracts (question_text, options, primitive_type, meta_dict) from diverse question item representations."""
    question_text = ""
    options = []
    primitive_type = "choice"
    meta_dict = None

    if isinstance(item, tuple):
        if len(item) == 4:
            question_text, options, primitive_type, meta_dict = item
        elif len(item) == 3:
            if isinstance(item[2], dict):
                question_text, options, meta_dict = item
                primitive_type = meta_dict.get("primitive_type", meta_dict.get("type", "choice"))
            else:
                question_text, options, primitive_type = item
                meta_dict = None
        elif len(item) == 2:
            question_text, options = item
            if isinstance(options, str) and options.lower() in ("noul", "choice", "score", "multichoice", "range", "extract"):
                primitive_type = options.lower()
                options = []
            else:
                primitive_type = "choice"
            meta_dict = None
        elif len(item) == 1:
            question_text = item[0]
            options = []
            primitive_type = "choice"
            meta_dict = None
    elif isinstance(item, dict):
        question_text = item.get("question", item.get("query", item.get("proposition", "")))
        options = item.get("options", item.get("levels", []))
        primitive_type = item.get("primitive_type", item.get("type", "choice"))
        meta_dict = item.get("metadata", item.get("meta", None))
    elif hasattr(item, "to_tuple") and callable(item.to_tuple):
        tup = item.to_tuple()
        if isinstance(tup, tuple):
            if len(tup) == 4:
                question_text, options, primitive_type, meta_dict = tup
            elif len(tup) == 3:
                question_text, options, primitive_type = tup
                meta_dict = getattr(item, "metadata", None)
            elif len(tup) == 2:
                question_text, options = tup
                primitive_type = getattr(item, "primitive_type", getattr(item, "type", item.__class__.__name__.lower()))
                meta_dict = getattr(item, "metadata", None)
            else:
                question_text = str(tup)
                options = []
        else:
            question_text = str(tup)
            options = []
    elif hasattr(item, "question") or hasattr(item, "proposition") or hasattr(item, "query"):
        question_text = getattr(item, "question", getattr(item, "proposition", getattr(item, "query", "")))
        options = getattr(item, "options", getattr(item, "levels", []))
        primitive_type = getattr(item, "primitive_type", getattr(item, "type", item.__class__.__name__.lower()))
        meta_dict = getattr(item, "metadata", None)
    else:
        question_text = str(item)
        options = []

    # Normalize primitive_type
    if isinstance(primitive_type, str):
        primitive_type = primitive_type.strip().lower()
    else:
        primitive_type = "choice"

    # Normalize options
    if options is None:
        options = []
    elif not isinstance(options, list):
        options = list(options)
    options = [str(opt) for opt in options]

    return question_text, options, primitive_type, meta_dict


class RhenPacker:
    """
    Rhen State & Branch Packaging Engine.
    
    Provides structured token layout construction, hierarchical path depth calculation,
    and exact coordinate extraction for <|decide|> (query) and <|/opt|> (option candidates).
    """

    def __init__(self, tokenizer=None, max_depth: int = 16):
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.max_depth = max_depth

    def serialize_state(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        allow_latent_slots: bool = False,
    ) -> str:
        """
        Serializes state into text, sanitizing delimiters to prevent delimiter spoofing.
        
        Args:
            state: Dictionary, JSON string, or raw text.
            allow_latent_slots: If True, preserves <|latent_slot|> tokens for continuous grounding.
            
        Returns:
            Sanitized formatted state string.
        """
        if isinstance(state, (dict, list)):
            # Formatted JSON representation
            raw_text = json.dumps(state, indent=2, ensure_ascii=False)
        else:
            raw_text = str(state)

        return sanitize_delimiter_text(raw_text, allow_latent_slots=allow_latent_slots)

    def _encode_branch(
        self,
        item: Any,
        all_ids: List[int],
        depth_list: List[int],
        allow_latent_slots: bool = False,
        parent_branch_idx: Optional[int] = None,
        condition_option_idx: Optional[int] = None,
    ) -> Tuple[PackedBranchMeta, int]:
        question_text, options, primitive_type, meta_dict = _parse_question_item(item)

        branch_start_pos = len(all_ids)
        clean_question = sanitize_delimiter_text(str(question_text), allow_latent_slots=allow_latent_slots)
        clean_options = [sanitize_delimiter_text(opt, allow_latent_slots=allow_latent_slots) for opt in options]

        # Start branch with question token
        q_prefix = f"{QUESTION_TOKEN} {clean_question} "
        q_ids = self.tokenizer.encode(q_prefix)
        all_ids.extend(q_ids)
        depth_list.extend([1] * len(q_ids))

        opt_end_indices = []
        opt_start_indices = []
        for opt in clean_options:
            opt_str = f"{OPTION_START_TOKEN} {opt} {OPTION_END_TOKEN} "
            opt_ids = self.tokenizer.encode(opt_str)
            opt_start_indices.append(len(all_ids))
            all_ids.extend(opt_ids)
            depth_list.extend([2] * len(opt_ids))
            opt_end_token_id = self.tokenizer.encode(OPTION_END_TOKEN)[-1]
            opt_end_idx = len(all_ids) - 1
            for idx in range(len(all_ids) - 1, len(all_ids) - len(opt_ids) - 1, -1):
                if all_ids[idx] == opt_end_token_id:
                    opt_end_idx = idx
                    break
            opt_end_indices.append(opt_end_idx)

        # Determine decision / trigger token based on primitive type
        if meta_dict and "trigger_token" in meta_dict:
            decide_str = str(meta_dict["trigger_token"])
        elif primitive_type == "multichoice":
            decide_str = MULTICHOICE_TOKEN
        elif primitive_type == "range":
            decide_str = RANGE_TOKEN
        elif primitive_type == "extract":
            decide_str = EXTRACT_TOKEN
        else:
            decide_str = DECIDE_TOKEN

        decide_ids = self.tokenizer.encode(decide_str)
        all_ids.extend(decide_ids)
        depth_list.extend([1] * len(decide_ids))
        decide_idx = len(all_ids) - 1

        b_len = len(all_ids) - branch_start_pos
        meta = PackedBranchMeta(
            question_text=question_text,
            options=options,
            branch_token_len=b_len,
            decide_idx=decide_idx,
            option_end_indices=opt_end_indices,
            num_options=len(options),
            primitive_type=primitive_type,
            metadata=meta_dict,
            parent_branch_idx=parent_branch_idx,
            condition_option_idx=condition_option_idx,
            option_start_indices=opt_start_indices,
        )
        return meta, b_len

    def _pack_sequence(
        self,
        state_prompt: str,
        questions_and_options: List[Any],
        is_delta: bool = False,
        latent_tensors: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        allow_latent_slots: bool = False,
    ) -> PackedRhenSequence:
        """
        Internal sequence packing engine for shared state (or delta) and parallel decision branches.
        """
        # 1. Encode State Prefix
        state_ids = self.tokenizer.encode(state_prompt)
        state_len = len(state_ids)

        # Compute depth_ids for the tokenized state prefix using compute_structural_depth_ids (or zeros)
        state_depths: List[int] = []
        try:
            line_depths = compute_structural_depth_ids(state_prompt)
            lines = state_prompt.splitlines(keepends=True)
            for line, d in zip(lines, line_depths):
                toks = self.tokenizer.encode(line)
                state_depths.extend([d] * len(toks))
        except Exception:
            state_depths = []

        if len(state_depths) < state_len:
            state_depths.extend([0] * (state_len - len(state_depths)))
        elif len(state_depths) > state_len:
            state_depths = state_depths[:state_len]

        all_ids = list(state_ids)
        depth_list = list(state_depths)

        branch_lengths = []
        branch_metas = []

        # 2. Encode Each Branch
        for item in questions_and_options:
            meta, b_len = self._encode_branch(
                item=item,
                all_ids=all_ids,
                depth_list=depth_list,
                allow_latent_slots=allow_latent_slots,
            )
            branch_lengths.append(b_len)
            branch_metas.append(meta)

        # 3. Create Attention Mask using Rhen2DBranchMask
        from rhen.optim.rhen_mask import Rhen2DBranchMask
        attn_mask = Rhen2DBranchMask.create_mask(
            state_length=state_len,
            branch_lengths=branch_lengths,
            device=device,
            dtype=torch.float32,
        )

        # Locate latent slot token indices if present
        latent_slot_id = getattr(self.tokenizer, "latent_slot_token_id", None)
        if latent_slot_id is None:
            try:
                latent_slot_id = self.tokenizer.token_to_id(LATENT_SLOT_TOKEN)
            except Exception:
                latent_slot_id = None

        latent_slot_indices: Optional[List[int]] = None
        if latent_slot_id is not None:
            found_indices = [idx for idx, tid in enumerate(all_ids) if tid == latent_slot_id]
            if found_indices:
                latent_slot_indices = found_indices

        # Compute GLM-4 2D branch position coordinates: 0 for state prefix, 1..K for branch depth
        b_depth_list = [0] * state_len
        branch_depth_map = {}
        for b_idx, meta in enumerate(branch_metas):
            if meta.parent_branch_idx is None:
                d = 1
            else:
                d = branch_depth_map.get(meta.parent_branch_idx, 1) + 1
            branch_depth_map[b_idx] = d
            b_depth_list.extend([d] * meta.branch_token_len)
        branch_depth_tensor = torch.tensor([b_depth_list], dtype=torch.long, device=device)

        input_ids_tensor = torch.tensor([all_ids], dtype=torch.long, device=device)

        return PackedRhenSequence(
            input_ids=input_ids_tensor,
            attention_mask=attn_mask,
            state_length=state_len,
            branch_lengths=branch_lengths,
            branches=branch_metas,
            depth_ids=depth_list,
            branch_depth_ids=branch_depth_tensor,
            is_delta=is_delta,
            latent_tensors=latent_tensors,
            latent_slot_indices=latent_slot_indices,
            is_speculative_tree=False,
        )

    def pack(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions_and_options: Optional[List[Any]] = None,
        latent_tensors: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        questions: Optional[List[Any]] = None,
        allow_latent_slots: Optional[bool] = None,
        **kwargs: Any,
    ) -> PackedRhenSequence:
        """
        Packs shared state and multiple parallel decision branches into a single tensor layout.

        Layout format:
        [STATE_PREFIX]
          <|state|> {serialized_state}
        [BRANCH_1]
          <|q|> {question_1} <|opt|> {opt_1_1} <|/opt|> <|opt|> {opt_1_2} <|/opt|> ... <|decide|>
        [BRANCH_2]
          <|q|> {question_2} <|opt|> {opt_2_1} <|/opt|> ... <|decide|>

        Args:
            state: Shared context / state object.
            questions_and_options: List of questions, passed as:
                - 4-tuples: `(question_text, options, primitive_type, meta_dict)`
                - 3-tuples: `(question_text, options, primitive_type)`
                - 2-tuples: `(question_text, options)`
                - Question objects/dicts with question, options, primitive_type, and metadata attributes.
                If primitive_type is 'range', options may be empty or contain [str(min_val), str(max_val)].
                If primitive_type is 'extract', question_text is query/slot name and options may be empty.
            latent_tensors: Optional dictionary of continuous feature tensors to ground into latent slots.
            device: Target torch device.
            questions: Alias for questions_and_options for backward compatibility.
            allow_latent_slots: Explicit opt-in to keep literal <|latent_slot|> tokens in the state
                and question text. Latent tensors imply it; text alone never does.

        Returns:
            PackedRhenSequence containing input_ids, 2D attention mask, and metadata.
        """
        if questions_and_options is None:
            questions_and_options = questions if questions is not None else []

        # Latent slots are enabled only by trusted arguments: continuous latent tensors or an
        # explicit allow_latent_slots=True. Literal <|latent_slot|> text in the state or in the
        # questions never enables them; it is neutralized unless one of those is given.
        allow_latent_slots = bool(allow_latent_slots) or latent_tensors is not None

        state_str = self.serialize_state(state, allow_latent_slots=allow_latent_slots)
        state_prompt = f"{STATE_TOKEN} {state_str}\n"

        if latent_tensors is not None and LATENT_SLOT_TOKEN not in state_str and not any(LATENT_SLOT_TOKEN in str(q) for q in questions_and_options):
            num_slots = 0
            for v in latent_tensors.values():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 3:
                        num_slots += v.shape[1]
                    elif v.dim() == 2:
                        num_slots += v.shape[0] if (v.shape[0] > 1 and v.shape[1] > 32) else 1
                    else:
                        num_slots += 1
                else:
                    num_slots += 1
            if num_slots > 0:
                slot_str = " ".join([LATENT_SLOT_TOKEN] * num_slots)
                state_prompt = f"{STATE_TOKEN} {state_str} {slot_str}\n"

        packed = self._pack_sequence(
            state_prompt=state_prompt,
            questions_and_options=questions_and_options,
            is_delta=False,
            latent_tensors=latent_tensors,
            device=device,
            allow_latent_slots=allow_latent_slots,
        )

        # Placeholders in the text (which may come from untrusted content) must match the
        # trusted latent tensors exactly, so content cannot add or drop latent slots.
        if latent_tensors is not None:
            expected_slots = 0
            for v in latent_tensors.values():
                if isinstance(v, torch.Tensor) and v.dim() == 3:
                    expected_slots += v.shape[1]
                elif isinstance(v, torch.Tensor) and v.dim() == 2:
                    expected_slots += v.shape[0] if (v.shape[0] > 1 and v.shape[1] > 32) else 1
                else:
                    expected_slots += 1
            found_slots = len(packed.latent_slot_indices or [])
            if found_slots != expected_slots:
                raise ValueError(
                    f"latent_tensors provide {expected_slots} slot(s) but the packed sequence contains "
                    f"{found_slots} <|latent_slot|> token(s); the placeholders must match the tensors"
                )
        return packed

    def pack_delta(
        self,
        base_state: Any,
        delta_state: Any,
        questions: List[Any],
        latent_tensors: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> PackedRhenSequence:
        """
        Packs a state delta against a base state using <|state_diff|> delta encoding.

        Format:
        f"{STATE_DIFF_TOKEN} BASE_HASH:{sha256(str(base_state))[:16]}\nDELTA:\n{delta_text}\n"
        (sha256 = hashlib.sha256(...).hexdigest(), stable across processes)

        Args:
            base_state: Base state object or representation.
            delta_state: Delta / incremental state updates.
            questions: List of decision questions and candidate options.
            latent_tensors: Optional dictionary of continuous feature tensors to ground into latent slots.
            device: Target torch device.

        Returns:
            PackedRhenSequence with is_delta=True and depth_ids computed.
        """
        # Same trusted-only rule as pack(): latent tensors enable slots, literal text does not.
        allow_latent_slots = latent_tensors is not None

        delta_text = self.serialize_state(delta_state, allow_latent_slots=allow_latent_slots)
        # Process-independent digest: the built-in hash() is salted per process (PYTHONHASHSEED).
        base_hash = hashlib.sha256(str(base_state).encode("utf-8")).hexdigest()[:16]
        state_prompt = f"{STATE_DIFF_TOKEN} BASE_HASH:{base_hash}\nDELTA:\n{delta_text}\n"

        if latent_tensors is not None and LATENT_SLOT_TOKEN not in delta_text and not any(LATENT_SLOT_TOKEN in str(q) for q in questions):
            num_slots = 0
            for v in latent_tensors.values():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 3:
                        num_slots += v.shape[1]
                    elif v.dim() == 2:
                        num_slots += v.shape[0] if (v.shape[0] > 1 and v.shape[1] > 32) else 1
                    else:
                        num_slots += 1
                else:
                    num_slots += 1
            if num_slots > 0:
                slot_str = " ".join([LATENT_SLOT_TOKEN] * num_slots)
                state_prompt = state_prompt.rstrip() + f" {slot_str}\n"

        return self._pack_sequence(
            state_prompt=state_prompt,
            questions_and_options=questions,
            is_delta=True,
            latent_tensors=latent_tensors,
            device=device,
            allow_latent_slots=allow_latent_slots,
        )

    def build_speculative_tree_pack(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        root_question: Any,
        speculative_branches: Dict[int, List[Any]],
        latent_tensors: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> PackedRhenSequence:
        """
        Builds a single packed sequence containing the root question PLUS all speculative child branches
        for each option of the root question.
        
        Constructs a 2D Tree-Block-Causal Mask where:
        * State prefix attends to state prefix.
        * Root question attends to state prefix + root question.
        * Child branch for option k attends to state prefix + root question + itself,
          but is BLOCKED from attending to other competing child branches!
        
        This allows all cascade stages across all paths to be evaluated in ONE parallel forward pass!
        
        Args:
            state: Shared context or environment state.
            root_question: Root stage question specification (tuple, dict, or object).
            speculative_branches: Mapping from option index k of root question to a list of child questions.
            latent_tensors: Optional dictionary of continuous feature tensors to ground into latent slots.
            device: Target torch device.
            
        Returns:
            PackedRhenSequence with is_speculative_tree=True.
        """
        # Same trusted-only rule as pack(): latent tensors enable slots, literal text does not.
        allow_latent_slots = latent_tensors is not None

        state_str = self.serialize_state(state, allow_latent_slots=allow_latent_slots)
        state_prompt = f"{STATE_TOKEN} {state_str}\n"

        if latent_tensors is not None and LATENT_SLOT_TOKEN not in state_str and not (LATENT_SLOT_TOKEN in str(root_question)):
            num_slots = 0
            for v in latent_tensors.values():
                if isinstance(v, torch.Tensor):
                    if v.dim() == 3:
                        num_slots += v.shape[1]
                    elif v.dim() == 2:
                        num_slots += v.shape[0] if (v.shape[0] > 1 and v.shape[1] > 32) else 1
                    else:
                        num_slots += 1
                else:
                    num_slots += 1
            if num_slots > 0:
                slot_str = " ".join([LATENT_SLOT_TOKEN] * num_slots)
                state_prompt = f"{STATE_TOKEN} {state_str} {slot_str}\n"

        state_ids = self.tokenizer.encode(state_prompt)
        state_len = len(state_ids)

        state_depths: List[int] = []
        try:
            line_depths = compute_structural_depth_ids(state_prompt)
            lines = state_prompt.splitlines(keepends=True)
            for line, d in zip(lines, line_depths):
                toks = self.tokenizer.encode(line)
                state_depths.extend([d] * len(toks))
        except Exception:
            state_depths = []

        if len(state_depths) < state_len:
            state_depths.extend([0] * (state_len - len(state_depths)))
        elif len(state_depths) > state_len:
            state_depths = state_depths[:state_len]

        all_ids = list(state_ids)
        depth_list = list(state_depths)

        branch_lengths: List[int] = []
        branch_metas: List[PackedBranchMeta] = []

        # 1. Encode Root Question (Branch 0)
        root_meta, root_len = self._encode_branch(
            item=root_question,
            all_ids=all_ids,
            depth_list=depth_list,
            allow_latent_slots=allow_latent_slots,
            parent_branch_idx=None,
            condition_option_idx=None,
        )
        branch_lengths.append(root_len)
        branch_metas.append(root_meta)

        # 2. Encode Speculative Child Branches for each option k
        def sort_key(k):
            try:
                return (0, int(k))
            except (ValueError, TypeError):
                return (1, str(k))

        sorted_option_keys = sorted(speculative_branches.keys(), key=sort_key)

        for opt_key in sorted_option_keys:
            child_items = speculative_branches[opt_key]
            if not isinstance(child_items, list):
                child_items = [child_items]

            try:
                cond_idx = int(opt_key)
            except (ValueError, TypeError):
                cond_idx = opt_key

            for child_q in child_items:
                c_meta, c_len = self._encode_branch(
                    item=child_q,
                    all_ids=all_ids,
                    depth_list=depth_list,
                    allow_latent_slots=allow_latent_slots,
                    parent_branch_idx=0,
                    condition_option_idx=cond_idx,
                )
                branch_lengths.append(c_len)
                branch_metas.append(c_meta)

        # 3. Construct 2D Tree-Block-Causal Mask
        total_len = len(all_ids)
        mask = torch.full((total_len, total_len), float("-inf"), device=device, dtype=torch.float32)

        # State prefix attends to state prefix (bidirectional)
        if state_len > 0:
            mask[:state_len, :state_len] = 0.0

        branch_spans: List[Tuple[int, int]] = []
        curr = state_len
        for b_len in branch_lengths:
            branch_spans.append((curr, curr + b_len))
            curr += b_len

        # Root question (Branch 0)
        root_start, root_end = branch_spans[0]
        root_len = branch_lengths[0]

        # Root question attends to state prefix
        if state_len > 0:
            mask[root_start:root_end, :state_len] = 0.0

        # Root question attends to itself causally
        root_causal = torch.triu(
            torch.full((root_len, root_len), float("-inf"), device=device, dtype=torch.float32),
            diagonal=1,
        )
        mask[root_start:root_end, root_start:root_end] = root_causal

        # Speculative child branches (Branch idx >= 1)
        for b_idx in range(1, len(branch_lengths)):
            b_start, b_end = branch_spans[b_idx]
            b_len = branch_lengths[b_idx]
            b_meta = branch_metas[b_idx]

            # Child branch attends to state prefix
            if state_len > 0:
                mask[b_start:b_end, :state_len] = 0.0

            # Child branch attends to root question
            mask[b_start:b_end, root_start:root_end] = 0.0

            # If child has intermediate ancestors along the path
            curr_parent = b_meta.parent_branch_idx
            while curr_parent is not None and curr_parent > 0:
                p_start, p_end = branch_spans[curr_parent]
                mask[b_start:b_end, p_start:p_end] = 0.0
                curr_parent = branch_metas[curr_parent].parent_branch_idx

            # Child branch attends to itself causally
            b_causal = torch.triu(
                torch.full((b_len, b_len), float("-inf"), device=device, dtype=torch.float32),
                diagonal=1,
            )
            mask[b_start:b_end, b_start:b_end] = b_causal
            # NOTE: Competing child branches for other options remain strictly -inf!

        attn_mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, total_len, total_len)

        # Locate latent slot token indices if present
        latent_slot_id = getattr(self.tokenizer, "latent_slot_token_id", None)
        if latent_slot_id is None:
            try:
                latent_slot_id = self.tokenizer.token_to_id(LATENT_SLOT_TOKEN)
            except Exception:
                latent_slot_id = None

        latent_slot_indices: Optional[List[int]] = None
        if latent_slot_id is not None:
            found_indices = [idx for idx, tid in enumerate(all_ids) if tid == latent_slot_id]
            if found_indices:
                latent_slot_indices = found_indices

        # Compute GLM-4 2D branch position coordinates: 0 for state prefix, 1..K for branch depth
        b_depth_list = [0] * state_len
        branch_depth_map = {}
        for b_idx, meta in enumerate(branch_metas):
            if meta.parent_branch_idx is None:
                d = 1
            else:
                d = branch_depth_map.get(meta.parent_branch_idx, 1) + 1
            branch_depth_map[b_idx] = d
            b_depth_list.extend([d] * meta.branch_token_len)
        branch_depth_tensor = torch.tensor([b_depth_list], dtype=torch.long, device=device)

        input_ids_tensor = torch.tensor([all_ids], dtype=torch.long, device=device)

        return PackedRhenSequence(
            input_ids=input_ids_tensor,
            attention_mask=attn_mask,
            state_length=state_len,
            branch_lengths=branch_lengths,
            branches=branch_metas,
            depth_ids=depth_list,
            branch_depth_ids=branch_depth_tensor,
            is_delta=False,
            latent_tensors=latent_tensors,
            latent_slot_indices=latent_slot_indices,
            is_speculative_tree=True,
        )


# Global default packer singleton
_GLOBAL_PACKER: Optional[RhenPacker] = None

def get_rhen_packer(tokenizer=None) -> RhenPacker:
    """Returns singleton RhenPacker instance."""
    global _GLOBAL_PACKER
    if _GLOBAL_PACKER is None or tokenizer is not None:
        _GLOBAL_PACKER = RhenPacker(tokenizer=tokenizer)
    return _GLOBAL_PACKER
