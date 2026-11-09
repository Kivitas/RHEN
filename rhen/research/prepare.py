"""Tokenize a text corpus into the flat token files used for language-model training.

    python -m rhen.research.prepare data/corpus raw/*.txt docs.jsonl --val-fraction 0.01

Inputs may be ``.txt`` files (one document per file), directories (every ``.txt`` and
``.jsonl`` file inside, recursively) or ``.jsonl`` files with a ``text`` field per line.
Documents are tokenized with the bundled EOS tokenizer unless ``--tokenizer`` names another.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterator, List

from rhen.research.data import prepare_token_corpus


def iter_documents(paths: List[str], text_field: str = "text") -> Iterator[str]:
    for path in paths:
        if os.path.isdir(path):
            children = []
            for root, _, files in os.walk(path):
                children += [os.path.join(root, f) for f in files if f.endswith((".txt", ".jsonl"))]
            yield from iter_documents(sorted(children), text_field)
        elif path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        text = json.loads(line).get(text_field)
                        if text:
                            yield str(text)
        else:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            if text.strip():
                yield text


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Prepare a token corpus for rhen.research.train.")
    parser.add_argument("out_dir")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args(argv)

    from tokenizer import DEFAULT_TOKENIZER_DIR, get_tokenizer

    tok = get_tokenizer(args.tokenizer or DEFAULT_TOKENIZER_DIR)
    meta = prepare_token_corpus(
        iter_documents(args.inputs, args.text_field),
        encode=lambda text: tok.encode(text, add_special_tokens=False),
        out_dir=args.out_dir,
        vocab_size=tok.vocab_size,
        eos_token_id=tok.eos_token_id,
        val_fraction=args.val_fraction,
        source_note=", ".join(args.inputs),
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
