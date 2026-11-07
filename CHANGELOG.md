# Changelog

All notable changes to **RHEN** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-11-07

### Added
- **Initial standalone open-source release of the RHEN decision engine.**
- **The 7 Typed Semantic Primitives**:
  - `Noul`: Binary semantic proposition evaluation ($P(\text{True}) \in [0, 1]$).
  - `Choice`: Categorical 1-of-$K$ selection with normalized simplex probabilities.
  - `Score`: Ordered continuous spectrum over ordinal levels.
  - `MultiChoice`: Independent subset selection against dynamic thresholds.
  - `Range`: Continuous interval regression via fitted Beta distributions.
  - `Extract`: Non-autoregressive extractive span pointers over state tokens.
  - `Cascade`: Multi-stage conditional decision sequences with speculative branch evaluation.
- **Ergonomic Functional API**:
  - `rhen.decide()` and `rhen.evaluate()` with automatic context normalization from raw dictionaries, lists, and JSON strings.
  - Process-level cached default engine via `get_default_engine()`.
- **Bidirectional Serialization**:
  - `to_dict()`, `to_json()`, `from_dict()`, `from_json()` across all 7 primitive results and `ActionCard`.
  - Automatic handling of PyTorch tensors, NumPy types, sets, and nested dataclasses.
- **Universal Backbone Adapters**:
  - `HuggingFaceEncoderAdapter` to attach RHEN's pointer heads to any HuggingFace model (BERT, RoBERTa, DeBERTa, LLaMA, Mistral, Qwen, etc.).
  - Configurable span pooling strategies: `"mean"`, `"last"`, `"cls"`.
  - `MockEncoder` for zero-download local testing and mocking.
- **Statistical Safety & Abstention**:
  - Split-conformal prediction sets ($1 - \alpha$ coverage guarantees).
  - Adaptive conformal martingale updates under non-stationary covariate shift.
  - Helmholtz Free Energy OOD detection ($E > 4.5$).
  - Geometric convex-hull selective abstention (`ANSWER`, `ABSTAIN`, `NEED_CLARIFICATION`).
- **Cryptographic Action Fabric**:
  - HMAC-SHA256 signed `ActionCard`s with canonical RFC-8785 JSON digests.
  - Risk tiers (`low`, `medium`, `high`, `critical`).
  - Dual-custody co-signing protocol and multi-agent BFT consensus.
  - Merkle action ledger with inclusion proofs under 0.1ms per leaf.
  - Out-of-band audit ledger streaming to isolated socket sinks.
- **Cognitive Extensions**:
  - RLCD: Reinforcement Learning for Calibrated Decisions with overconfidence penalties.
  - Duration Thresholding (TDT): Adaptive recurrent loop reduction.
  - Hierarchical Choice Router: Coarse-to-fine clustering for 77+ options (scoring $<25\%$ of options).
  - OOD Regression Guard and Evidence Verification Lattice.
  - Multilingual Decision Router across 10 languages.
- **Command-Line Interface (CLI)**:
  - `rhen-decide`: CLI decision evaluation with piped stdin support.
  - `rhen-train`: Model training and fine-tuning.
  - `rhen-eval`: Evaluation metrics, ECE calibration, and OOD AUROC.
  - `rhen-tasks`: Synthetic benchmark dataset generation across 6 task families.
- **Examples & Framework Integrations**:
  - 7 runnable examples covering quickstart, action cards, conformal prediction, HuggingFace adapters, hierarchical routing, empirical JEV latency benchmarks, and LangChain/CrewAI/FastAPI recipes.
