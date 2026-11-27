# Changelog

All notable changes to **RHEN** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **RHEN-2 (`rhen.v2`, work in progress)**. Design in `docs/RHEN2_ARCHITECTURE_PLAN.md`; nothing is trained and no accuracy, latency or calibration result is claimed.
  - Cached-State Deliberation Lattice model: state encoder run once and cacheable, deliberation workspace with weight-shared loops, option-set layer (order-invariant by construction), typed readouts (choice, noul, score, multi, range, extract) with a trained null option.
  - Packing (`Rhen2Packer`, `QuestionSpec`, `Rhen2Example`), losses, and the Checker (correctness head, evidence spans, interventions, rule checks, verdicts).
  - Calibration layer (temperatures, isotonic, group conformal sets, OOD) and a unified decision pipeline (`DecisionPipeline`) with an expected-loss policy and a fail-closed rule: without calibration nothing above the `low` tier executes.
  - Outcome ledger for feedback and drift alarms; routing for option sets above 255 options.
  - Data: 24 dataset converters, 47 synthetic task families, pretraining builders, a benchmark blocklist (`rhen/v2/data/blocklist.json`), and teacher labelling.
  - Trainer and CLI: `rhen2-train` and `rhen2-eval`; configs in `configs/v2/`.
  - Evaluation suite (`benchmarks/suite/`), ablation script (`run_ablations.py`) and latency script (`benchmarks/latency/`). Flat-encoder baseline with parameter and FLOP matching and a comparison runner (`flat_vs_workspace.py`); the tiny-scale result is in `docs/FLAT_BASELINE_RESULTS.md` (flat baseline more accurate in distribution, workspace exactly option-order invariant).
  - Presets `r2-tiny`, `r2-50m`, `r2-300m`, `r2-1b` and `r2-8b`, with exact parameter counts from `rhen.v2.config.count_parameters`.
- **Effort tiers & gated intra-branch option isolation**:
  - Added configurable compute effort tiers (`low`, `medium`, `high`) to scale recurrent deliberation depth at inference time.
  - Added gated intra-branch option isolation (`isolate_branch_options`) so candidate options within a decision branch can be masked from cross-attending to one another to strengthen option-order invariance.
- **Banking77 decision corpus generator & L4 GPU training config**:
  - Added `bank77corpus.py` (backed by `benchmarks/banking77/bank77corpus.py`) to build stratified `train.jsonl`, `val.jsonl`, `test.jsonl`, `test_ood.jsonl`, and `corpus_manifest.json` (with SHA-256 digests and exact training/evaluation commands) in `full77`, `hard_negatives`, and `mixed` modes, plus an offline `--synthetic` smoke mode.
  - Added `configs/banking77_l4_gpu.json`, a `TrainConfig` tuned for single-GPU NVIDIA L4 (24 GB VRAM, `bf16`, `small` preset, Muon + WSD schedule, option-order consistency).
- **Predictive layer** (`rhen/forecasting.py`):
  - `Forecast`: median path plus nine quantiles (levels 0.1 to 0.9) per step, with `interval()`, `prob_exceeds()`, `copula_breach_prob()` (stationary AR(1) Gaussian-copula horizon breach probability clamped within Fréchet-Hoeffding bounds), and `uncertainty(scale_floor=...)` computed from the quantile function.
  - `NaiveQuantileForecaster`: dependency-free random-walk baseline and default forecaster.
  - `TimesFMForecaster`: adapter for TimesFM 3.0 (`timesfm3`). It requires `license_acknowledged=True` because the 3.0 weights are non-commercial only.
  - Error metrics: `pinball_loss`, `interval_coverage`, `mase`, `brier_score`.
- **Forecast-aware decisions** (`rhen/decision_pipeline.py`):
  - `ForecastAwareDecider`: forecasts series, conditions `RhenEngine.choice` on the forecast summary, surfaces `breach_prob_copula` alongside lower/upper breach bounds in decision features, applies the policy, and gates execution through `approve` and `execute`.
  - `DecisionPolicy`: deterministic gate returning `EXECUTE`, `NEEDS_APPROVAL`, `ABSTAIN` or `BLOCKED`, with reasons.
  - `FeedbackLedger`: records decisions and outcomes, reports calibration (pinball loss, coverage, Brier), and round-trips through JSONL.
- `forecast` optional extra: `pip install "rhen[forecast]"` (installs `timesfm[torch]>=3.0.2,<4`).
- Example `examples/08_forecast_aware_decisions.py` (baseline forecaster by default; TimesFM only with `--timesfm --accept-noncommercial-license`).
- Tests in `tests/test_forecast_integration.py` and `tests/test_bank77corpus_and_forecast.py`. The real TimesFM test runs only with `RHEN_TIMESFM_RUN_REAL=1` and a cached checkpoint.
- README section "Predictive layer and forecast-aware decisions".

### Changed
- **Legacy config keys emit a `FutureWarning`**. Python shows it under its default filters, whereas a `DeprecationWarning` is hidden from library callers. The warning names every removed key passed to `EOSConfig`, its group classes, `from_dict` or `from_pretrained`. Those keys are still accepted and dropped, so older configs and checkpoints load.

### Removed
- **BREAKING for code that reads these attributes directly.** The 21 deprecated no-op config fields below were removed in the release after 1.0.0. No code read them. Passing them as keywords or in a config dict still works, with the `FutureWarning` above. Reading them as attributes (for example `cfg.use_triton`) now raises `AttributeError`.
  - `RecurrentLoopConfig`: `enable_symplectic_loop`
  - `AttentionMLAConfig`: `use_flash_attn`, `diff_attn_lambda_init`, `enable_prefix_lm_mask`
  - `YaRNConfig`: `yarn_scale_factor`
  - `RhenEngineConfig`: `rhen_calib_loss_coef`, `rhen_enable_deep_reasoning`, `rhen_confidence_threshold`, `rhen_enable_qwen_reasoning`, `rhen_enable_deepseek_reasoning`, `rhen_meta_reasoning_cache_size`, `rhen_enable_three_layer_pipeline`, `rhen_semantic_layer_weight`, `rhen_reasoning_layer_weight`, `rhen_generation_layer_weight`, `rhen_enable_confidence_routing`, `rhen_min_semantic_confidence`, `rhen_max_reasoning_depth`
  - `OrchestratorConfig`: `use_fused_recurrent_kernels`, `checkpoint_policy`, `use_triton`

### Fixed
- **`forward_rhen` without a decision head.** `forward_rhen` now raises `RuntimeError` when `enable_rhen=False`, instead of returning random decisions.
- **Known v1 structural limits are documented by tests.** Four `xfail(strict=True)` tests in `tests/test_v1_known_structural_issues.py` record the v1 decision-path limits: the state is not bidirectional in odd layers, the slot channel mixes branches, positions make answers depend on order, and option isolation is opt-in. These are known limits, not fixes; `rhen.v2` provides the properties by construction.
- **Adapter mask order.** `HuggingFaceEncoderAdapter` again tries the packed 4D branch-isolation mask first. Any exception from that attempt is retried once with the 2D padding mask, as in 1.0.0. If the 2D attempt also fails, the original 4D error is raised, chained to the second. An encoder without an `attention_mask` keyword is still called without one (DEBUG log).
- **Mask downgrade logging.** The 4D-to-2D downgrade logs one WARNING per encoder instance, because branch isolation may not hold for that encoder.
- **Tokenizer fallback.** `SemanticBoundaryChunker.count_tokens` catches `Exception` again, so tokenizers that raise plain `Exception` subclasses fall back to the character estimate. The fallback logs one WARNING.
- **Other logged fallbacks.** The state-depth and latent-slot fallbacks in `rhen_packer` log one WARNING per site. `FusedSwiGLU` replaces `print()` with logging, and a failed compiled forward logs at DEBUG.
- **Checkpoint trust (`rhen-rng-v2`).** `optimizers.pt` and new `rhen-rng-v2` RNG state checkpoints serialize NumPy/Python/PyTorch RNG state as plain tensors and primitives so they load cleanly with `weights_only=True`. Legacy `rng_rank*.pt` checkpoints remain readable with a fallback warning; `load_checkpoint` documents that legacy checkpoints must come from a trusted source.
- **RLCD margin & PPO optimization.** Fixed `RLCDOptimizer` margin reward and PPO clipped surrogate computation so policy ratio updates and calibration penalties align cleanly during reinforcement learning.
- **Range variance calibration loss.** Fixed `Range` primitive loss to calibrate predictive Beta variance alongside mean accuracy rather than under-regularizing spread.
- **Router grouping.** `HierarchicalChoiceRouter` groups options by character-trigram similarity (balanced, deterministic k-medoids) instead of by position, and labels each group with its medoid. Its accuracy against a flat scorer is not measured in this repo.
- **Banking77 measurement.** `benchmarks/banking77/` reports baselines on the full 3,080-example test split: TF-IDF logistic regression 87.9% (2,707 correct), TF-IDF nearest centroid 80.8%, majority class 1.3%. The untrained RHEN variant was run on 3 examples only (0 of 3 correct). It is a pipeline check, not an accuracy estimate.
- **README claims.** Removed "production-grade" and "Mission-Critical" from the introduction, and removed "empirically calibrated" and "coverage guarantees" wording. Removed the unmeasured router claim "scores <25% of options without accuracy loss". Calibration, conformal coverage, router accuracy and the absence of comparisons with other systems are now listed under "Known limitations".

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
