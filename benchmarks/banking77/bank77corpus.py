#!/usr/bin/env python3
"""Banking77 decision corpus builder for RHEN training and evaluation.

Builds stratified ``train.jsonl``, ``val.jsonl``, ``test.jsonl``, ``test_ood.jsonl``,
and ``corpus_manifest.json`` in RHEN's ``DecisionExample`` JSONL format, ready for
training on an NVIDIA L4 GPU via ``rhen-train --config configs/banking77_l4_gpu.json``.

Reuses the zero-parquet dataset loader and local caching from
``benchmarks/banking77/run_eval.py``, and provides an offline ``--synthetic`` fallback
for smoke testing without internet access.

Modes:
  full77          Every example packs all 77 Banking77 intent labels (with optional
                  deterministic per-example option permutation to remove positional bias).
  hard_negatives  Training examples pack ``--num-options K`` options: the ground-truth
                  intent plus the top K-1 hardest TF-IDF nearest-centroid / lexical
                  neighbors. ``val.jsonl`` and ``test.jsonl`` retain all 77 options by
                  default (or K with ``--eval-match-k``).
  mixed           Combines hard-negative curriculum examples with full-77 examples and
                  auxiliary ``noul`` binary verification questions on the shared state.

Examples:
    python bank77corpus.py --out-dir data/banking77 --mode mixed
    python bank77corpus.py --out-dir data/banking77 --mode full77 --holdout-ood-intents 0
    python bank77corpus.py --synthetic --out-dir /tmp/b77_smoke --mode mixed
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def _load_run_eval_module():
    """Import sibling benchmarks/banking77/run_eval.py cleanly from any invocation style."""
    module_name = "_rhen_banking77_run_eval"
    if module_name in sys.modules:
        return sys.modules[module_name]
    run_eval_path = SCRIPT_DIR / "run_eval.py"
    spec = importlib.util.spec_from_file_location(module_name, run_eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {run_eval_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_run_eval = _load_run_eval_module()

DEFAULT_CACHE_DIR: Path = _run_eval.DEFAULT_CACHE_DIR
HF_DATASET: str = _run_eval.HF_DATASET
RHEN_QUESTION: str = _run_eval.RHEN_QUESTION
TfidfVectorizer = _run_eval.TfidfVectorizer
load_banking77 = _run_eval.load_banking77

DEFAULT_OUT_DIR = Path("data/banking77")
MODES: Tuple[str, ...] = ("full77", "hard_negatives", "mixed")

# Canonical 77 intent labels of PolyAI/banking77 in exact label-index order (0..76).
INTENT_LABELS: Tuple[str, ...] = (
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)

# Out-of-domain financial & general customer queries used when --holdout-ood-intents=0.
OOD_QUERIES: Tuple[Tuple[str, str], ...] = (
    ("mortgage", "What is the current 30-year fixed mortgage refinancing rate for a duplex?"),
    ("mortgage", "Can I remove private mortgage insurance once my home equity hits twenty percent?"),
    ("tax", "How do I report capital gains from selling employer stock options on Form 8949?"),
    ("tax", "What is the deadline for filing a quarterly estimated self-employment tax return?"),
    ("tax", "Can I deduct home office internet expenses on Schedule C this tax season?"),
    ("investing", "Should I rebalance my retirement portfolio between index ETFs and municipal bonds?"),
    ("investing", "How does dividend reinvestment affect my cost basis in a brokerage account?"),
    ("investing", "What is the expense ratio difference between S&P 500 mutual funds and ETFs?"),
    ("crypto", "How do I recover an Ethereum hardware wallet using a 24-word BIP39 seed phrase?"),
    ("crypto", "Why is my Layer-2 rollup bridge transaction stuck waiting for gas confirmation?"),
    ("insurance", "Does comprehensive auto insurance cover hail damage while parked at an airport?"),
    ("insurance", "How do I file a dental insurance reimbursement claim for an out-of-network crown?"),
    ("loan", "What credit score is needed to qualify for a federal student loan consolidation?"),
    ("loan", "Can I pay extra principal on my auto loan without triggering a prepayment penalty?"),
    ("business", "How do I register a single-member LLC and obtain an EIN from the IRS?"),
    ("business", "What accounting software integrates best with payroll tax filings for ten employees?"),
    ("travel", "My flight to Zurich was delayed four hours; am I eligible for EU261 compensation?"),
    ("travel", "What is the checked baggage weight limit for international economy tickets?"),
    ("weather", "Will it rain in Seattle tomorrow afternoon during the marathon?"),
    ("weather", "What is the UV index forecast for Phoenix this weekend?"),
    ("tech", "How do I fix a CUDA out-of-memory error when fine-tuning a transformer in PyTorch?"),
    ("tech", "Why does my Docker container fail to resolve DNS names on a corporate VPN?"),
    ("ecommerce", "The blender I ordered arrived with a cracked pitcher; how do I print a return label?"),
    ("ecommerce", "Can I change the shipping address on my online furniture order before dispatch?"),
    ("healthcare", "What are the side effects of taking ibuprofen on an empty stomach?"),
    ("healthcare", "How do I transfer my prescription refill to a pharmacy closer to my office?"),
    ("legal", "Can my landlord raise the rent mid-lease if there is no escalation clause?"),
    ("legal", "How do I apostille a birth certificate for a work visa application abroad?"),
    ("general", "What is a good vegetarian lasagna recipe that freezes well for meal prep?"),
    ("general", "How long does it take to train for a half marathon starting from a 5K base?"),
    ("real_estate", "What closing costs are typically paid by the seller versus the buyer?"),
    ("retirement", "What is the annual contribution limit for a Roth IRA if I am over fifty?"),
)

_SYNTHETIC_TEMPLATES: Tuple[str, ...] = (
    "Hi, I need help with {phrase} on my account right now.",
    "Can you explain what is going on with {phrase}? I tried in the app already.",
    "I am having an issue regarding {phrase} and would like to resolve it today.",
    "Could someone from support assist me with {phrase}?",
    "Why am I seeing a problem related to {phrase}?",
    "Please check my banking profile for {phrase} as soon as possible.",
    "I noticed something unexpected about {phrase} earlier this morning.",
    "How do I handle {phrase} using the mobile banking app?",
    "Is there any update on {phrase}? It has been worrying me.",
    "I tried to sort out {phrase}, can you guide me through the next steps?",
    "My app gave me an error while checking {phrase}, what should I do?",
    "Could you confirm the policy and status for {phrase}?",
    "I urgently need clarification about {phrase} for my recent activity.",
    "Whenever I attempt {phrase}, it does not behave as expected.",
)

_SYNTHETIC_CONTEXT_SNIPPETS: Tuple[str, ...] = (
    "in my app",
    "for my UK account",
    "after my last purchase",
    "since yesterday evening",
    "when travelling abroad",
    "on my personal card",
    "earlier this week",
    "during checkout",
    "on my statement",
    "for a GBP transaction",
)


def _label_to_phrase(label: str) -> str:
    """Turn a snake_case Banking77 label into readable English text."""
    return label.replace("_", " ").replace("?", "").strip()


def generate_synthetic_banking77(
    seed: int = 0,
    train_per_class: int = 10,
    test_per_class: int = 4,
    label_names: Sequence[str] = INTENT_LABELS,
) -> Dict[str, Any]:
    """Generate an offline synthetic Banking77-like dataset across all 77 intents.

    Deterministic for a given ``seed`` and produces zero-network smoke data with the
    same dictionary structure returned by ``load_banking77``.
    """
    if train_per_class < 2:
        raise ValueError(f"train_per_class must be >= 2 for stratified train/val split, got {train_per_class}")
    if test_per_class < 1:
        raise ValueError(f"test_per_class must be >= 1, got {test_per_class}")

    rng = np.random.default_rng(int(seed))
    labels_list = list(label_names)

    def _make_split(per_class: int, split_tag: str) -> Dict[str, Any]:
        texts: List[str] = []
        labels: List[int] = []
        for idx, name in enumerate(labels_list):
            phrase = _label_to_phrase(name)
            tokens = phrase.split()
            for j in range(per_class):
                tpl = _SYNTHETIC_TEMPLATES[int(rng.integers(0, len(_SYNTHETIC_TEMPLATES)))]
                ctx = _SYNTHETIC_CONTEXT_SNIPPETS[int(rng.integers(0, len(_SYNTHETIC_CONTEXT_SNIPPETS)))]
                # Slight deterministic variation so train and test strings differ
                if j % 3 == 0:
                    text = tpl.format(phrase=phrase)
                elif j % 3 == 1:
                    text = f"{tpl.format(phrase=phrase)} Specifically {ctx} ({split_tag} ref {idx}-{j})."
                else:
                    emph = " ".join(tokens[: max(1, len(tokens) - 1)])
                    text = f"Question about {phrase}: I keep running into {emph} issues {ctx}."
                texts.append(text)
                labels.append(idx)
        order = rng.permutation(len(texts))
        ordered_texts = [texts[int(i)] for i in order]
        ordered_labels = np.asarray([labels[int(i)] for i in order], dtype=np.int64)
        return {
            "texts": ordered_texts,
            "labels": ordered_labels,
            "label_names": labels_list,
            "file": None,
            "sha256": None,
            "rows": len(ordered_texts),
            "num_rows_total": len(ordered_texts),
        }

    return {
        "loader": "synthetic-offline",
        "hf_dataset": "synthetic/banking77",
        "hf_revision": None,
        "label_names": labels_list,
        "splits": {
            "train": _make_split(train_per_class, "tr"),
            "test": _make_split(test_per_class, "te"),
        },
    }


def stratified_train_val_split(
    texts: Sequence[str],
    labels: np.ndarray,
    val_fraction: float = 0.10,
    seed: int = 0,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Deterministically split examples into stratified train and validation slices."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    rng = np.random.default_rng(int(seed))
    by_label: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(labels.tolist()):
        by_label[int(y)].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for label in sorted(by_label):
        idxs = np.asarray(by_label[label], dtype=np.int64)
        perm = rng.permutation(idxs)
        n = len(perm)
        if n <= 1:
            train_idx.extend(perm.tolist())
            continue
        n_val = max(1, int(round(n * val_fraction)))
        n_val = min(n_val, n - 1)  # always leave at least 1 example in train
        val_idx.extend(perm[:n_val].tolist())
        train_idx.extend(perm[n_val:].tolist())

    train_perm = rng.permutation(np.asarray(train_idx, dtype=np.int64))
    val_perm = rng.permutation(np.asarray(val_idx, dtype=np.int64))
    train_texts = [texts[int(i)] for i in train_perm]
    val_texts = [texts[int(i)] for i in val_perm]
    return train_texts, labels[train_perm], val_texts, labels[val_perm]


def _char_trigrams(text: str) -> set[str]:
    cleaned = f"  {text.lower().replace('_', ' ').strip()}  "
    if len(cleaned) < 3:
        return {cleaned}
    return {cleaned[i : i + 3] for i in range(len(cleaned) - 2)}


class HardNegativeMiner:
    """TF-IDF nearest-centroid + lexical neighbor miner for hard-negative option packing."""

    def __init__(
        self,
        train_texts: Sequence[str],
        train_labels: np.ndarray,
        label_names: Sequence[str],
    ) -> None:
        self.label_names = list(label_names)
        self.n_classes = len(self.label_names)
        label_phrases = [_label_to_phrase(name) for name in self.label_names]

        # Fit TF-IDF on training texts + label descriptions so every label has non-zero representation.
        min_df = 2 if len(train_texts) >= 200 else 1
        self.vectorizer = TfidfVectorizer(min_df=min_df).fit(list(train_texts) + label_phrases)
        X_train = self.vectorizer.transform(train_texts)
        X_labels = self.vectorizer.transform(label_phrases)

        centroids = np.zeros((self.n_classes, X_train.shape[1]), dtype=np.float32)
        counts = np.zeros(self.n_classes, dtype=np.float32)
        for i, y in enumerate(train_labels.tolist()):
            centroids[int(y)] += X_train[i]
            counts[int(y)] += 1.0

        # Blend document centroid with label-phrase TF-IDF (fallback to label phrase when unseen)
        for c in range(self.n_classes):
            if counts[c] > 0:
                doc_c = centroids[c] / counts[c]
                norm = float(np.linalg.norm(doc_c))
                if norm > 1e-12:
                    doc_c = doc_c / norm
                centroids[c] = 0.85 * doc_c + 0.15 * X_labels[c]
            else:
                centroids[c] = X_labels[c]

        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self.centroids = centroids / norms
        self.centroid_sim = self.centroids @ self.centroids.T

        # Precompute character-trigram Jaccard similarity between label names for lexical neighbors
        trigrams = [_char_trigrams(p) for p in label_phrases]
        lex_sim = np.zeros((self.n_classes, self.n_classes), dtype=np.float32)
        for i in range(self.n_classes):
            ti = trigrams[i]
            for j in range(i, self.n_classes):
                tj = trigrams[j]
                union = len(ti | tj)
                score = (len(ti & tj) / union) if union else 0.0
                lex_sim[i, j] = score
                lex_sim[j, i] = score
        self.lexical_sim = lex_sim

    def mine_batch(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        num_negatives: int,
        allowed_negatives: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        """Return the top ``num_negatives`` hardest negative class indices for each example."""
        if num_negatives < 1:
            return [[] for _ in texts]
        X = self.vectorizer.transform(texts)
        query_scores = X @ self.centroids.T  # [N, C]

        pool_mask: Optional[np.ndarray] = None
        if allowed_negatives is not None:
            pool_mask = np.zeros(self.n_classes, dtype=bool)
            for idx in allowed_negatives:
                pool_mask[int(idx)] = True

        hard_negs: List[List[int]] = []
        for i, y in enumerate(labels.tolist()):
            y_int = int(y)
            # Combine per-query centroid cosine, inter-class centroid cosine, and lexical similarity
            scores = (
                0.60 * query_scores[i]
                + 0.25 * self.centroid_sim[y_int]
                + 0.15 * self.lexical_sim[y_int]
            ).copy()
            if pool_mask is not None:
                scores[~pool_mask] = -np.inf
            scores[y_int] = -np.inf
            # Stable descending order (ties broken deterministically by class index)
            ranked = sorted(
                (c for c in range(self.n_classes) if np.isfinite(scores[c])),
                key=lambda c: (-float(scores[c]), c),
            )
            hard_negs.append(ranked[:num_negatives])
        return hard_negs


def _make_choice_question(
    options: List[str],
    true_label_name: str,
    rng: np.random.Generator,
    permute_options: bool,
    question: str = RHEN_QUESTION,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    opts = list(options)
    if permute_options:
        perm = rng.permutation(len(opts))
        opts = [opts[int(i)] for i in perm]
    label_idx = opts.index(true_label_name)
    q: Dict[str, Any] = {
        "question": question,
        "options": opts,
        "primitive": "choice",
        "label": int(label_idx),
    }
    if metadata:
        q["metadata"] = dict(metadata)
    return q


def _make_noul_question(label_name: str, is_positive: bool) -> Dict[str, Any]:
    return {
        "question": f"Is the customer's banking intent '{label_name}'?",
        "options": ["Yes", "No"],
        "primitive": "noul",
        "label": bool(is_positive),
        "metadata": {"candidate_intent": label_name, "verification": True},
    }


def _write_jsonl_with_sha256(records: Sequence[Dict[str, Any]], path: Path) -> Tuple[str, int, Dict[str, int], Dict[str, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    num_questions = 0
    primitives: Counter[str] = Counter()
    label_dist: Counter[str] = Counter()

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            line = json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n"
            fh.write(line)
            hasher.update(line.encode("utf-8"))
            for q in rec["questions"]:
                num_questions += 1
                prim = str(q["primitive"])
                primitives[prim] += 1
                if prim == "choice":
                    meta = q.get("metadata") or {}
                    if meta.get("is_ood") and meta.get("ood_type") == "out_of_domain_query":
                        label_dist[f"ood:{meta.get('ood_domain', 'unknown')}"] += 1
                    else:
                        label_dist[q["options"][int(q["label"])]] += 1

    return hasher.hexdigest(), num_questions, dict(sorted(primitives.items())), dict(sorted(label_dist.items()))


def build_corpus(
    out_dir: Path | str = DEFAULT_OUT_DIR,
    mode: str = "mixed",
    num_options: int = 16,
    val_fraction: float = 0.10,
    holdout_ood_intents: int = 7,
    permute_options: bool = True,
    eval_match_k: bool = False,
    mixed_full_ratio: float = 0.5,
    seed: int = 0,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    synthetic: bool = False,
    synthetic_train_per_class: int = 10,
    synthetic_test_per_class: int = 4,
    prefer_datasets_library: bool = True,
) -> Dict[str, Any]:
    """Build Banking77 RHEN decision JSONL splits and ``corpus_manifest.json``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if not 0.0 <= mixed_full_ratio <= 1.0:
        raise ValueError(f"mixed_full_ratio must be in [0, 1], got {mixed_full_ratio}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if synthetic:
        data = generate_synthetic_banking77(
            seed=seed,
            train_per_class=synthetic_train_per_class,
            test_per_class=synthetic_test_per_class,
        )
    else:
        data = load_banking77(Path(cache_dir), prefer_datasets_library=prefer_datasets_library)

    label_names: List[str] = list(data["label_names"])
    n_classes = len(label_names)
    if holdout_ood_intents < 0 or holdout_ood_intents >= n_classes:
        raise ValueError(
            f"holdout_ood_intents must be in [0, {n_classes - 1}], got {holdout_ood_intents}"
        )
    if mode in ("hard_negatives", "mixed"):
        if num_options < 2 or num_options > n_classes:
            raise ValueError(f"num_options must be in [2, {n_classes}], got {num_options}")

    raw_train_texts: List[str] = list(data["splits"]["train"]["texts"])
    raw_train_labels: np.ndarray = np.asarray(data["splits"]["train"]["labels"], dtype=np.int64)
    raw_test_texts: List[str] = list(data["splits"]["test"]["texts"])
    raw_test_labels: np.ndarray = np.asarray(data["splits"]["test"]["labels"], dtype=np.int64)

    rng = np.random.default_rng(int(seed))

    # Select held-out OOD intents (if requested)
    if holdout_ood_intents > 0:
        ood_intent_indices = sorted(
            int(i) for i in rng.choice(n_classes, size=holdout_ood_intents, replace=False)
        )
    else:
        ood_intent_indices = []
    ood_intent_set = set(ood_intent_indices)
    id_intent_indices = [i for i in range(n_classes) if i not in ood_intent_set]

    # Filter training pool to ID intents, then stratify into train/val
    id_train_mask = np.asarray([int(y) not in ood_intent_set for y in raw_train_labels], dtype=bool)
    id_train_texts = [t for t, keep in zip(raw_train_texts, id_train_mask) if keep]
    id_train_labels = raw_train_labels[id_train_mask]
    train_texts, train_labels, val_texts, val_labels = stratified_train_val_split(
        id_train_texts, id_train_labels, val_fraction=val_fraction, seed=seed + 11
    )

    # Filter official test split into ID test and (if holdout > 0) unseen-intent OOD test
    id_test_mask = np.asarray([int(y) not in ood_intent_set for y in raw_test_labels], dtype=bool)
    test_texts = [t for t, keep in zip(raw_test_texts, id_test_mask) if keep]
    test_labels = raw_test_labels[id_test_mask]

    ood_test_texts = [t for t, keep in zip(raw_test_texts, id_test_mask) if not keep]
    ood_test_labels = raw_test_labels[~id_test_mask]

    # Fit hard-negative miner strictly on training split (zero leakage from val/test)
    miner: Optional[HardNegativeMiner] = None
    if mode in ("hard_negatives", "mixed"):
        miner = HardNegativeMiner(train_texts, train_labels, label_names)

    # ---- Build train.jsonl ----
    train_records: List[Dict[str, Any]] = []
    train_hard_negs: Optional[List[List[int]]] = None
    if miner is not None:
        # Mine hardest negatives from all labels (at least num_options - 1, or 1 for noul)
        k_mine = max(num_options - 1, 1)
        train_hard_negs = miner.mine_batch(train_texts, train_labels, num_negatives=k_mine)

    # In mixed mode, decide deterministically which training examples use full77 vs hard_negatives
    mixed_use_full = (
        rng.random(len(train_texts)) < mixed_full_ratio if mode == "mixed" else np.zeros(len(train_texts), dtype=bool)
    )
    if mode == "mixed" and len(train_texts) >= 2:
        # Ensure at least one hard-negative and one full-77 example in small smoke corpora
        if mixed_full_ratio > 0.0 and not bool(mixed_use_full.any()):
            mixed_use_full[0] = True
        if mixed_full_ratio < 1.0 and bool(mixed_use_full.all()):
            mixed_use_full[-1] = False

    for i, (text, y_raw) in enumerate(zip(train_texts, train_labels.tolist())):
        y = int(y_raw)
        true_name = label_names[y]

        if mode == "full77":
            q = _make_choice_question(
                label_names,
                true_name,
                rng=rng,
                permute_options=permute_options,
                metadata={"intent_id": y, "mode": "full77"},
            )
            train_records.append(
                {"state": text, "questions": [q], "split": "train", "task": "banking77"}
            )
        elif mode == "hard_negatives":
            assert train_hard_negs is not None
            neg_ids = train_hard_negs[i][: num_options - 1]
            # Sort alphabetically before optional permutation so ground truth is never fixed at slot 0
            opts = sorted([true_name] + [label_names[c] for c in neg_ids])
            q = _make_choice_question(
                opts,
                true_name,
                rng=rng,
                permute_options=permute_options,
                metadata={"intent_id": y, "mode": "hard_negatives", "num_options": len(opts)},
            )
            train_records.append(
                {"state": text, "questions": [q], "split": "train", "task": "banking77"}
            )
        else:  # mode == "mixed"
            assert train_hard_negs is not None
            neg_ids = train_hard_negs[i]
            if bool(mixed_use_full[i]):
                choice_q = _make_choice_question(
                    label_names,
                    true_name,
                    rng=rng,
                    permute_options=permute_options,
                    metadata={"intent_id": y, "mode": "mixed_full77"},
                )
            else:
                opts = sorted([true_name] + [label_names[c] for c in neg_ids[: num_options - 1]])
                choice_q = _make_choice_question(
                    opts,
                    true_name,
                    rng=rng,
                    permute_options=permute_options,
                    metadata={"intent_id": y, "mode": "mixed_hard_negatives", "num_options": len(opts)},
                )
            hardest_neg_name = label_names[neg_ids[0]]
            pos_noul = _make_noul_question(true_name, is_positive=True)
            neg_noul = _make_noul_question(hardest_neg_name, is_positive=False)
            train_records.append(
                {
                    "state": text,
                    "questions": [choice_q, pos_noul, neg_noul],
                    "split": "train",
                    "task": "banking77_mixed",
                }
            )

    # ---- Helper for val / test splits ----
    def _build_eval_records(
        texts_seq: Sequence[str],
        labels_arr: np.ndarray,
        split_name: str,
        is_ood_split: bool = False,
    ) -> List[Dict[str, Any]]:
        use_k = mode == "hard_negatives" and eval_match_k and miner is not None
        hard_negs_eval = (
            miner.mine_batch(texts_seq, labels_arr, num_negatives=num_options - 1) if use_k else None
        )
        recs: List[Dict[str, Any]] = []
        for idx_i, (txt, y_val) in enumerate(zip(texts_seq, labels_arr.tolist())):
            yi = int(y_val)
            tname = label_names[yi]
            if use_k and hard_negs_eval is not None:
                opts = sorted([tname] + [label_names[c] for c in hard_negs_eval[idx_i][: num_options - 1]])
            else:
                opts = list(label_names)
            meta: Dict[str, Any] = {"intent_id": yi, "is_ood": is_ood_split}
            if is_ood_split:
                meta["ood_type"] = "held_out_intent"
            q = _make_choice_question(
                opts,
                tname,
                rng=rng,
                permute_options=False,
                metadata=meta,
            )
            recs.append(
                {
                    "state": txt,
                    "questions": [q],
                    "split": split_name,
                    "task": "banking77_ood" if is_ood_split else "banking77",
                }
            )
        return recs

    val_records = _build_eval_records(val_texts, val_labels, split_name="val", is_ood_split=False)
    test_records = _build_eval_records(test_texts, test_labels, split_name="test", is_ood_split=False)

    # ---- Build test_ood.jsonl ----
    if holdout_ood_intents > 0:
        ood_records = _build_eval_records(ood_test_texts, ood_test_labels, split_name="test_ood", is_ood_split=True)
        ood_strategy = "held_out_intents"
    else:
        # All 77 intents are in train/val; emit off-domain general/financial OOD queries
        ood_records = []
        eval_opts = list(label_names)
        for q_idx, (domain, query_text) in enumerate(OOD_QUERIES):
            ood_records.append(
                {
                    "state": query_text,
                    "questions": [
                        {
                            "question": RHEN_QUESTION,
                            "options": eval_opts,
                            "primitive": "choice",
                            # Valid placeholder index for DecisionQuestion schema; OOD AUROC reads free_energy & max_prob.
                            "label": int(q_idx % len(eval_opts)),
                            "metadata": {
                                "is_ood": True,
                                "ood_type": "out_of_domain_query",
                                "ood_domain": domain,
                            },
                        }
                    ],
                    "split": "test_ood",
                    "task": "banking77_ood",
                }
            )
        ood_strategy = "out_of_domain_queries"

    # ---- Write splits & compute SHA-256 digests ----
    split_payloads = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
        "test_ood": ood_records,
    }
    split_manifest: Dict[str, Any] = {}
    for split_name, recs in split_payloads.items():
        file_name = f"{split_name}.jsonl"
        file_path = out_path / file_name
        sha, n_q, prims, dist = _write_jsonl_with_sha256(recs, file_path)
        split_manifest[split_name] = {
            "file": file_name,
            "path": file_path.as_posix(),
            "num_examples": len(recs),
            "num_questions": n_q,
            "primitives": prims,
            "sha256": sha,
            "label_distribution": dist,
        }

    out_posix = out_path.as_posix()
    default_posix = DEFAULT_OUT_DIR.as_posix()
    if out_posix == default_posix:
        train_cmd = "rhen-train --config configs/banking77_l4_gpu.json"
    else:
        train_cmd = (
            f"rhen-train --config configs/banking77_l4_gpu.json "
            f"--decision-train {out_posix}/train.jsonl --decision-val {out_posix}/val.jsonl"
        )
    eval_cmd = (
        f"rhen-eval --checkpoint runs/banking77_l4/checkpoints/step_00003000 "
        f"--decision-dir {out_posix} --device cuda --precision bf16"
    )

    manifest: Dict[str, Any] = {
        "dataset": data["hf_dataset"],
        "hf_revision": data.get("hf_revision"),
        "loader": data["loader"],
        "synthetic": bool(synthetic),
        "mode": mode,
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "num_options": int(num_options) if mode in ("hard_negatives", "mixed") else n_classes,
        "permute_options": bool(permute_options),
        "eval_match_k": bool(eval_match_k),
        "mixed_full_ratio": float(mixed_full_ratio) if mode == "mixed" else None,
        "holdout_ood_intents": int(holdout_ood_intents),
        "ood_strategy": ood_strategy,
        "ood_intents": [label_names[i] for i in ood_intent_indices],
        "id_intents": [label_names[i] for i in id_intent_indices],
        "num_labels_total": n_classes,
        "question": RHEN_QUESTION,
        "splits": split_manifest,
        "commands": {
            "dry_run": f"{train_cmd} --dry-run",
            "train_l4_gpu": train_cmd,
            "eval_l4_gpu": eval_cmd,
        },
    }

    manifest_path = out_path / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bank77corpus.py",
        description="Build Banking77 RHEN decision JSONL corpora (train/val/test/test_ood) for L4 GPU training.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for JSONL splits and corpus_manifest.json (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for downloaded Banking77 JSON splits (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default="mixed",
        help="Option packing strategy: full77, hard_negatives, or mixed (default: mixed).",
    )
    parser.add_argument(
        "--num-options",
        type=int,
        default=16,
        help="Number of options per training example in hard_negatives/mixed modes (default: 16).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of in-distribution training examples reserved for stratified val.jsonl (default: 0.10).",
    )
    parser.add_argument(
        "--holdout-ood-intents",
        type=int,
        default=7,
        help="Number of Banking77 intents held out from train/val into test_ood.jsonl. "
        "Pass 0 to keep all 77 intents in train/val and generate off-domain OOD queries (default: 7).",
    )
    parser.add_argument(
        "--permute-options",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministically permute option order per training example to prevent positional overfitting (default: True).",
    )
    parser.add_argument(
        "--eval-match-k",
        action="store_true",
        help="In hard_negatives mode, also pack --num-options K in val/test instead of all 77 intents.",
    )
    parser.add_argument(
        "--mixed-full-ratio",
        type=float,
        default=0.5,
        help="Fraction of training choice questions using all 77 options in mixed mode (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for splits, OOD intent selection, permutation, and synthetic generation (default: 0).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate an offline synthetic 77-intent smoke corpus without internet access.",
    )
    parser.add_argument(
        "--synthetic-train-per-class",
        type=int,
        default=10,
        help="Number of synthetic train+val examples per intent when --synthetic is set (default: 10).",
    )
    parser.add_argument(
        "--synthetic-test-per-class",
        type=int,
        default=4,
        help="Number of synthetic test examples per intent when --synthetic is set (default: 4).",
    )
    parser.add_argument(
        "--prefer-rows-api",
        action="store_true",
        help="Bypass the HuggingFace datasets library and fetch via the zero-parquet rows API.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = build_corpus(
        out_dir=args.out_dir,
        mode=args.mode,
        num_options=args.num_options,
        val_fraction=args.val_fraction,
        holdout_ood_intents=args.holdout_ood_intents,
        permute_options=args.permute_options,
        eval_match_k=args.eval_match_k,
        mixed_full_ratio=args.mixed_full_ratio,
        seed=args.seed,
        cache_dir=args.cache_dir,
        synthetic=args.synthetic,
        synthetic_train_per_class=args.synthetic_train_per_class,
        synthetic_test_per_class=args.synthetic_test_per_class,
        prefer_datasets_library=not args.prefer_rows_api,
    )
    splits = manifest["splits"]
    print(f"Built Banking77 RHEN corpus ({manifest['mode']}) in {args.out_dir}:")
    for name in ("train", "val", "test", "test_ood"):
        sp = splits[name]
        print(
            f"  {sp['file']:<14} examples={sp['num_examples']:<6} "
            f"questions={sp['num_questions']:<6} sha256={sp['sha256'][:12]}..."
        )
    print(f"  manifest       {(Path(args.out_dir) / 'corpus_manifest.json').as_posix()}")
    print("\nNext steps (NVIDIA L4 GPU):")
    print(f"  Train : {manifest['commands']['train_l4_gpu']}")
    print(f"  Eval  : {manifest['commands']['eval_l4_gpu']}")
    return 0


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_OUT_DIR",
    "HF_DATASET",
    "INTENT_LABELS",
    "MODES",
    "OOD_QUERIES",
    "RHEN_QUESTION",
    "HardNegativeMiner",
    "TfidfVectorizer",
    "build_corpus",
    "build_parser",
    "generate_synthetic_banking77",
    "load_banking77",
    "main",
    "stratified_train_val_split",
]


if __name__ == "__main__":
    sys.exit(main())
