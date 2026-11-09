import logging
import os
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer

logger = logging.getLogger("tokenizer")

#: The bundled tokenizer directory (packaged as rhen/tokenizer_assets), resolved next to
#: this module so it works from any CWD.
_this_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZER_DIR = os.path.join(_this_dir, "tokenizer_assets")

STATE_TOKEN = "<|state|>"
STATE_DIFF_TOKEN = "<|state_diff|>"
QUESTION_TOKEN = "<|q|>"
OPTION_START_TOKEN = "<|opt|>"
OPTION_END_TOKEN = "<|/opt|>"
DECIDE_TOKEN = "<|decide|>"
MULTICHOICE_TOKEN = "<|multichoice|>"
RANGE_TOKEN = "<|range|>"
EXTRACT_TOKEN = "<|extract|>"
LATENT_SLOT_TOKEN = "<|latent_slot|>"

RHEN_SPECIAL_TOKENS = [
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
]

class EOSTokenizer:
    """
    EOS Tokenizer with full Rhen System One Decision Delimiter Registry.

    Loads a tokenizer from a local directory (default: the tokenizer_assets directory next to
    this module). Remote code is never executed (trust_remote_code=False) and nothing is
    downloaded: a path that does not exist, or cannot be loaded, raises RuntimeError.
    """
    def __init__(self, pretrained_model_name_or_path: str = DEFAULT_TOKENIZER_DIR):
        path = pretrained_model_name_or_path
        if not os.path.exists(path):
            raise RuntimeError(
                f"EOS tokenizer path '{path}' does not exist. Pass a local tokenizer directory "
                "(default: the tokenizer_assets directory next to this module, or one written by save_pretrained or "
                "build_custom_bpe_tokenizer). No remote download is attempted."
            )

        logger.info("Loading EOS Tokenizer from '%s'.", path)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=False,
                padding_side="right"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load EOS tokenizer from '{path}': {e}") from e

        # Register special delimiter tokens for Rhen if not present
        special_tokens_dict = {"additional_special_tokens": []}
        existing_special = set(self.tokenizer.all_special_tokens) if hasattr(self.tokenizer, "all_special_tokens") else set()
        
        for token in RHEN_SPECIAL_TOKENS:
            if token not in existing_special:
                special_tokens_dict["additional_special_tokens"].append(token)
                
        if special_tokens_dict["additional_special_tokens"]:
            self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    @property
    def state_diff_token_id(self) -> int:
        return self.token_to_id(STATE_DIFF_TOKEN)

    @property
    def state_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|state|>")

    @property
    def q_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|q|>")

    @property
    def opt_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|opt|>")

    @property
    def opt_end_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|/opt|>")

    @property
    def decide_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|decide|>")

    @property
    def multichoice_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|multichoice|>")

    @property
    def range_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|range|>")

    @property
    def extract_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("<|extract|>")

    @property
    def latent_slot_token_id(self) -> int:
        return self.token_to_id(LATENT_SLOT_TOKEN)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def build_rhen_decision_prompt(self, state: str, questions: List[Dict[str, Any]]) -> str:
        """
        Builds a structured prompt formatted with Rhen delimiter tokens:
        <|state|> {state} <|q|> {question_1} <|opt|> {opt_1} <|/opt|> ... <|decide|> ...
        """
        prompt_parts = ["<|state|>", state.strip()]
        for q in questions:
            q_text = q.get("question", "").strip()
            options = q.get("options", [])
            prompt_parts.append("<|q|>")
            prompt_parts.append(q_text)
            for opt in options:
                prompt_parts.append("<|opt|>")
                prompt_parts.append(str(opt).strip())
                prompt_parts.append("<|/opt|>")
            q_type = q.get("primitive_type", q.get("type", "choice"))
            if q_type == "multichoice":
                prompt_parts.append(MULTICHOICE_TOKEN)
            elif q_type == "range":
                prompt_parts.append(RANGE_TOKEN)
            elif q_type == "extract":
                prompt_parts.append(EXTRACT_TOKEN)
            else:
                prompt_parts.append(DECIDE_TOKEN)
        return " ".join(prompt_parts)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory)
        logger.info("EOS Tokenizer saved to '%s'.", save_directory)


# Alias for backward compatibility
EOS100kTokenizer = EOSTokenizer

_GLOBAL_TOKENIZER: Optional[EOSTokenizer] = None

def get_tokenizer(pretrained_model_name_or_path: str = DEFAULT_TOKENIZER_DIR) -> EOSTokenizer:
    """Returns singleton EOSTokenizer instance."""
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        _GLOBAL_TOKENIZER = EOSTokenizer(pretrained_model_name_or_path)
    return _GLOBAL_TOKENIZER


def build_custom_bpe_tokenizer(
    text_corpus_files: List[str],
    vocab_size: int = 50257,
    save_path: str = DEFAULT_TOKENIZER_DIR
) -> EOSTokenizer:
    """
    Trains a custom BPE tokenizer using HuggingFace Tokenizers and registers Rhen delimiters.

    Failures are logged and re-raised; there is no silent fallback to another tokenizer.
    """
    try:
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
        tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        special_tokens = ["<|endoftext|>", "<|pad|>"] + RHEN_SPECIAL_TOKENS
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        tokenizer.train(text_corpus_files, trainer)
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save(os.path.join(save_path, "tokenizer.json"))
        return EOSTokenizer(save_path)
    except Exception:
        logger.exception(
            "Custom BPE tokenizer build failed (corpus=%s, save_path='%s'); re-raising.",
            text_corpus_files, save_path,
        )
        raise


RhenTokenizer = EOSTokenizer
get_rhen_tokenizer = get_tokenizer

