# Changelog

All notable changes to **RHEN** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Predictive layer** (`rhen/forecasting.py`):
  - `Forecast`: median path plus nine quantiles (levels 0.1 to 0.9) per step, with `interval()`, `prob_exceeds()` and `uncertainty()` computed from the quantile function.
  - `NaiveQuantileForecaster`: dependency-free random-walk baseline and default forecaster.
  - `TimesFMForecaster`: adapter for TimesFM 3.0 (`timesfm3`). It requires `license_acknowledged=True` because the 3.0 weights are non-commercial only.
  - Error metrics: `pinball_loss`, `interval_coverage`, `mase`, `brier_score`.
- **Forecast-aware decisions** (`rhen/decision_pipeline.py`):
  - `ForecastAwareDecider`: forecasts series, conditions `RhenEngine.choice` on the forecast summary, applies the policy, and gates execution through `approve` and `execute`.
  - `DecisionPolicy`: deterministic gate returning `EXECUTE`, `NEEDS_APPROVAL`, `ABSTAIN` or `BLOCKED`, with reasons.
  - `FeedbackLedger`: records decisions and outcomes, reports calibration (pinball loss, coverage, Brier), and round-trips through JSONL.
- `forecast` optional extra: `pip install "rhen[forecast]"` (installs `timesfm[torch]>=3.0.2,<4`).
- Example `examples/08_forecast_aware_decisions.py` (baseline forecaster by default; TimesFM only with `--timesfm --accept-noncommercial-license`).
- Tests in `tests/test_forecast_integration.py`. The real TimesFM test runs only with `RHEN_TIMESFM_RUN_REAL=1` and a cached checkpoint.
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
- **Adapter mask order.** `HuggingFaceEncoderAdapter` again tries the packed 4D branch-isolation mask first. Any exception from that attempt is retried once with the 2D padding mask, as in 1.0.0. If the 2D attempt also fails, the original 4D error is raised, chained to the second. An encoder without an `attention_mask` keyword is still called without one (DEBUG log).
- **Mask downgrade logging.** The 4D-to-2D downgrade logs one WARNING per encoder instance, because branch isolation may not hold for that encoder.
- **Tokenizer fallback.** `SemanticBoundaryChunker.count_tokens` catches `Exception` again, so tokenizers that raise plain `Exception` subclasses fall back to the character estimate. The fallback logs one WARNING.
- **Other logged fallbacks.** The state-depth and latent-slot fallbacks in `rhen_packer` log one WARNING per site. `FusedSwiGLU` replaces `print()` with logging, and a failed compiled forward logs at DEBUG.
- **Checkpoint trust.** `optimizers.pt` loads with `weights_only=True`. `rng_rank*.pt` still loads with `weights_only=False` and logs a WARNING, because NumPy RNG state needs full unpickling. `load_checkpoint` documents that checkpoints must come from a trusted source.
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
