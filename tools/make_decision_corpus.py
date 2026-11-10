"""Write a deterministic language-model corpus of decision-making text.

    python tools/make_decision_corpus.py --out data/corpus_decision_1b

Each example from ``rhen.research.tasks`` (lookup, routing, approval, risk, comparison, checks and
the other families) is rendered as plain text: the state, the question, the lettered options and
the correct answer. About 40 examples are packed into each document. The labels are exact, so the
text shows what a correct decision looks like. It is still generated text: the rules are simple
and the phrasings repeat. It is not a corpus of real decisions.

Examples are drawn in chunks, so memory stays flat. Chunk seeds are fixed, so the same arguments
always produce byte-identical files. The default produces about 1.0 billion tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rhen.research.data import DecisionExample, prepare_token_corpus  # noqa: E402
from rhen.research.tasks import generate_split  # noqa: E402
from rhen.tokenizer import DEFAULT_TOKENIZER_DIR, get_tokenizer  # noqa: E402

CHUNK = 60_000        # divisible by the 6 task families, so every chunk is balanced
GENERATOR_FILE = "generator.json"


def render(ex: DecisionExample) -> str:
    """One decision example as text: the state, then each question with its options and answer."""
    lines = [f"State: {json.dumps(ex.state, sort_keys=True)}"]
    for q in ex.questions:
        lines.append(f"Question: {q.question}")
        if q.primitive == "noul":
            lines.append("Answer: " + ("yes" if q.label else "no"))
            continue
        opts = " ".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(q.options))
        lines.append(f"Options: {opts}")
        if q.primitive == "choice":
            lab = int(q.label)
            lines.append(f"Answer: ({chr(65 + lab)}) {q.options[lab]}")
        elif q.primitive == "multichoice":
            picked = ", ".join(f"({chr(65 + i)}) {q.options[i]}" for i in sorted(q.label))
            lines.append(f"Answer: {picked}")
        else:  # score
            lines.append(f"Answer: level {int(round(float(q.label)))}")
    return "\n".join(lines)


def _complete_with_same_parameters(out_dir: str, params: dict) -> bool:
    try:
        with open(os.path.join(out_dir, GENERATOR_FILE), encoding="utf-8") as fh:
            if json.load(fh) != params:
                return False
        with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        train = os.path.getsize(os.path.join(out_dir, "train.bin"))
        val = os.path.getsize(os.path.join(out_dir, "val.bin"))
    except (OSError, ValueError):
        return False
    return train == 2 * meta["train_tokens"] and val == 2 * meta["val_tokens"]


def build(out_dir: str, examples: int, per_doc: int, seed: int) -> dict:
    tok = get_tokenizer(DEFAULT_TOKENIZER_DIR)
    chunks = examples // CHUNK

    def documents():
        buf = []
        for k in range(chunks):
            for ex in generate_split("train", CHUNK, f"{seed}:{k}"):
                buf.append(render(ex))
                if len(buf) == per_doc:
                    yield "\n\n".join(buf)
                    buf = []
            if (k + 1) % 20 == 0 or k + 1 == chunks:
                print(f"generated {(k + 1) * CHUNK} examples", flush=True)
        if buf:
            yield "\n\n".join(buf)

    return prepare_token_corpus(
        documents(),
        encode=lambda text: tok.encode(text, add_special_tokens=False),
        out_dir=out_dir, vocab_size=tok.vocab_size, eos_token_id=tok.eos_token_id,
        val_fraction=0.01,
        source_note=f"decision-making text; examples={chunks * CHUNK} per_doc={per_doc} seed={seed}",
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/corpus_decision_1b")
    parser.add_argument("--examples", type=int, default=282 * CHUNK,
                        help="rounded down to a multiple of 60000 (default 16,920,000)")
    parser.add_argument("--per-doc", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    examples = (args.examples // CHUNK) * CHUNK
    params = {"examples": examples, "per_doc": args.per_doc, "seed": args.seed, "format": "decision-text-v1"}
    if not args.force and _complete_with_same_parameters(args.out, params):
        with open(os.path.join(args.out, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        print(json.dumps({"out": os.path.abspath(args.out), "status": "already complete, skipped",
                          "total_tokens": meta["train_tokens"] + meta["val_tokens"]}, indent=2))
        return

    sidecar = os.path.join(args.out, GENERATOR_FILE)
    if os.path.exists(sidecar):
        os.remove(sidecar)
    meta = build(args.out, examples, args.per_doc, args.seed)
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
    print(json.dumps({"out": os.path.abspath(args.out), "status": "built",
                      "total_tokens": meta["train_tokens"] + meta["val_tokens"],
                      "content_sha256": meta.get("content_sha256")}, indent=2))


if __name__ == "__main__":
    main()
