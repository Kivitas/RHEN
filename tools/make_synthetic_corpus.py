"""Write a deterministic synthetic language-model corpus in the rhen.research token layout.

    python tools/make_synthetic_corpus.py --out data/corpus_synth_1b

Token ids are drawn from a Zipf-shaped distribution over the bundled EOS tokenizer's vocabulary,
in fixed-length documents. The output is NOT natural language: it exercises the training loop,
data loading, throughput and checkpointing. The same arguments always produce byte-identical
files, which ``meta.json`` records as ``content_sha256``.

The defaults (1,000,000 documents of 1,000 tokens, seed 20261110) give about 1.001 billion tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Make `rhen` importable when this file is run directly from a checkout, without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rhen.research.data import prepare_token_corpus  # noqa: E402
from rhen.tokenizer import DEFAULT_TOKENIZER_DIR, get_tokenizer  # noqa: E402


def build(out_dir: str, docs: int, doc_tokens: int, seed: int, zipf_exponent: float) -> dict:
    tok = get_tokenizer(DEFAULT_TOKENIZER_DIR)
    vocab = tok.vocab_size
    eos = tok.eos_token_id
    ids = np.array([i for i in range(vocab) if i != eos], dtype=np.int64)
    weights = 1.0 / np.arange(1, len(ids) + 1, dtype=np.float64) ** zipf_exponent
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    rng = np.random.default_rng(seed)

    def texts():
        for i in range(docs):
            yield f"synthetic-doc-{i}"  # only used to derive the stable train/val split

    def encode(_text: str):
        u = rng.random(doc_tokens)
        return ids[np.searchsorted(cdf, u, side="right").clip(max=len(ids) - 1)].tolist()

    return prepare_token_corpus(
        texts(), encode=encode, out_dir=out_dir, vocab_size=vocab, eos_token_id=eos,
        val_fraction=0.01,
        source_note=f"synthetic Zipf token ids (not natural text); docs={docs} doc_tokens={doc_tokens} seed={seed}",
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/corpus_synth_1b")
    parser.add_argument("--docs", type=int, default=1_000_000)
    parser.add_argument("--doc-tokens", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--zipf-exponent", type=float, default=1.07)
    args = parser.parse_args(argv)
    meta = build(args.out, args.docs, args.doc_tokens, args.seed, args.zipf_exponent)
    total = meta["train_tokens"] + meta["val_tokens"]
    print(json.dumps({"out": os.path.abspath(args.out), "total_tokens": total,
                      "content_sha256": meta.get("content_sha256")}, indent=2))


if __name__ == "__main__":
    main()
