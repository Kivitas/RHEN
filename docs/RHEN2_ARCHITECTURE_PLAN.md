# RHEN-2 Architecture and Pipeline Plan

Status: PLAN ONLY. No code in this repository was changed by this document.
Date: 2026-11-12. Scope: deep scan of the current code, a new architecture
(RHEN-2), the decision pipeline, a built-in reasoner/checker, and the scale
ladder 50M -> 300M -> 1B -> 8B.

This document is written so that a smaller model (Sonnet or Haiku) can implement
it task by task without having to re-derive the reasoning. Read Section 0 first.

---

## 0. How to use this document (rules for implementers)

1. Work one task at a time, in the order of Section 11. Each task lists the
   files to create or edit, the exact behaviour, and an acceptance test. A task
   is done only when its acceptance test passes and `pytest tests/ -q` still
   passes.
2. Do not edit anything under `rhen/` that is not named in the task. The current
   code (called **v1** here) stays working as legacy until RHEN-2 passes the
   gates in Section 10. All new code goes in `rhen/v2/` and `tests/v2/`.
3. Never invent a number. Every threshold in the pipeline must come from a
   calibration file fitted on held-out data (Section 8). If no calibration file
   is loaded, the pipeline must say `calibrated: false` and must not auto-execute
   anything above the `low` risk tier.
4. Never claim a result in the README that is not backed by a committed JSON
   file under `benchmarks/`. The current README already follows this rule; keep it.
5. When a step says "shape", the tensor must have exactly that shape. Add an
   `assert` for it in the code. Letters used everywhere:
   - `B` examples in a batch, `M` questions per example, `K` options per question
   - `Ls` state tokens, `Lq` question tokens, `Lo` tokens per option
   - `d` model width, `R` deliberation loops actually run
6. If a task is ambiguous, stop and ask. Do not guess, and do not "improve" the
   design while implementing it.
7. Suggested model per task is marked `[Haiku]` (mechanical) or `[Sonnet]`
   (needs care). Tasks marked `[Review]` should be checked by a stronger model
   or a human before merge.

---

## 1. Summary

### 1.1 What the scan found

RHEN v1 is a causal language model (EOS: MLA attention, MoE, a looped core) with
a decision head bolted on through a block mask. The safety and audit tooling
around it is careful and honestly documented. The model path has structural
problems that training cannot fix:

- The properties the README promises (bidirectional state, isolated branches,
  order invariance) are **not** what the code computes. Three separate
  mechanisms break them (findings A1, A2, A3 in Section 3).
- The weights are pretrained under one attention pattern (causal) and used for
  decisions under another (bidirectional state, Rhen slot channels on). The
  decision-only modules receive no pretraining signal at all (A4).
- Decision training runs one example at a time and needs a second full forward
  pass per example just to push the model towards option-order invariance (A5, A6).
- Most of the parameter and code budget goes to features that serve token
  generation (KV-cache compression, multi-token prediction, speculative
  decoding), which a decision model never uses (A14).
- There is no trained checkpoint, no real decision dataset, and no measured
  calibration. The pipeline thresholds are hard-coded constants (C2).

### 1.2 What this plan proposes

A purpose-built decision architecture, **RHEN-2 "Cached-State Deliberation
Lattice" (CSDL)**, with five parts:

1. **State Encoder**: a bidirectional encoder that reads the state once. Its
   output is a cacheable memory.
2. **Deliberation Workspace**: a small weight-shared recurrent block that runs
   `R` loops over only the question and option tokens, reading the state memory
   by cross-attention. Compute is spent on thinking about the decision, not on
   re-reading the state.
3. **Option Set Layer**: options are a set, not a sequence. Order invariance is
   guaranteed by construction, so the second training pass disappears.
4. **Typed Readouts** with a built-in **null option** ("cannot tell") in every
   question.
5. **Checker**: a built-in reasoner that verifies each decision by evidence
   pointing, causal interventions on the state, depth consistency and a learned
   probability-of-correct head. It can confirm, ask for more loops, or escalate.

Then one unified decision pipeline replaces the four overlapping ones, with
calibrated thresholds, cost-based policy and per-group conformal sets.

### 1.3 Honest expectations

- "Comparable to JEV" cannot be promised by an architecture. JEV's edge, by its
  own founder's statement, is calibration training data, not architecture.
  Section 7 (data) matters more than Section 4 (architecture).
- CSDL is a new composition of known primitives (attention, recurrence,
  cross-attention, set layers). It is a different architecture from a GPT-style
  decoder and from a BERT-style encoder with a head, but it is not a new
  primitive. Its claimed advantages are hypotheses until the ablations in
  Section 10.4 are run.
- Realistic targets after training (targets, not predictions):

| Size | Realistic target |
| :--- | :--- |
| 50M | Matches fine-tuned Laya (421M) on narrow tasks after per-task fine-tuning; weak zero-shot. Best 50M will come from distilling the 1B model. |
| 300M | Laya-class zero-shot and better calibration; close to JEV on classification-style tasks. |
| 1B | JEV-class on most public classification and routing benchmarks; still behind on multi-hop and arithmetic. |
| 8B | Microsoft-Decision-1 class (that model is a post-trained 9B LLM). Needs roughly a trillion pretraining tokens or a pretrained backbone. |

---

## 2. Research findings

Confidence labels: **[P]** primary or vendor source read directly, **[V]** vendor
claim relayed by a third party, **[3P]** third-party analysis, **[S]** speculation.
None of the JEV internals are published. Treat every JEV architecture statement
as [S] unless marked otherwise.

### 2.1 JEV (TypeSafe AI)

- [V] "System One" model: returns typed values, no generated tokens. A "parallel
  sampler" reads the state once and answers every question in one forward pass.
- [V] API: one `state` (text, JSON or array of text) plus a `questions` map. Each
  question has `type`, `instructions`, `criteria`. Types: Choice (max 255
  options), Score (ordered criteria, continuous score), Noul (yes-probability).
  Questions are evaluated in parallel and in isolation.
- [V] Limits: 64k tokens for state plus questions; state plus the longest
  question must fit in 32k.
- [V] Confidence is "a statistic computed from the probability distribution";
  the statistic is not named. Noul has no separate confidence.
- [V] Training is called RLCD (Reinforcement Learning for Calibrated Decisions).
  No paper, no reward function and no scoring rule are published. The founder is
  quoted saying it is trained only on synthetic data and that the bottleneck is
  calibration data, not architecture.
- [S] Parameter count, encoder vs decoder, and weights are not disclosed. One
  analyst guesses it is close to Laya's design.
- [V] Latency 70 to 500 ms end to end; third parties measure 236 to 276 ms p50.
- [V/3P] Reported numbers: 67.8% on TypeSafe's four-workflow benchmark (GPT-class
  74.1%, Opus-class 73.1%). Third-party study: SST-2 96.4%, ARC 98.8%, CLINC150
  89.5% (151 options), Banking77 0.870, pooled ECE 0.028 over 22 Choice datasets
  with a per-dataset range of 0.003 to 0.279. Microsoft's table gives Jev 1.13.0
  82.3% accuracy over 36 benchmarks.
- [V] Documented weak spots ("jaggedness"): literal reading, arithmetic and
  counting, date comparison, multi-hop indirection, large noisy state,
  adversarial content. Complementary Noul pairs need not sum to 1 (one published
  example sums to 1.19). No explanation is returned.
- [V] Recommended use: act at high confidence, confirm at medium, human at low
  (example thresholds 0.5 and 0.9); keep policy and permissions in code.

Design lessons for RHEN-2: support up to 255 options natively; a 32k state at
the top scale; fix the published weak spots on purpose (numeric channel, Noul
consistency by construction, evidence output, null option); calibration data is
the main investment.

### 2.2 Laya (Convai Innovations, open weights, Apache-2.0)

- [V] ModernBERT-large encoder (about 395M) plus a decision head of two
  transformer layers, a type embedding per question type and a per-option scorer.
  421M total. Multilingual variant on mmBERT-base, 322M.
- [V] Input: `[CLS] <type> question [SEP] [MASK] option1 [MASK] option2 ... [SEP]
  state [SEP]`. Each option is scored at its `[MASK]`; softmax over the options.
- [V] Context 512 (English) to 1,024 tokens; up to 8,192 for multilingual.
- [V] Training: proper scoring rules (log, spherical, ranked probability score
  for ordinal). Temperature refit per (question type x option count).
- [V] ECE 0.081 after refit (0.466 before). Zero-shot is near random (0.362 vs
  0.318 random on its own benchmark); fine-tuned 0.766 vs Jev 0.727. Banking77:
  0.425 vs Jev 0.870. About 33 to 40 ms per question on a T4.
- [V] Limits: degrades past about 20 options; state capped at 512 tokens; weak
  at counting and arithmetic; one checkpoint scored 0.000 on Khmer at 0.952
  confidence.

Design lessons: a plain encoder with options in the same sequence is cheap but
breaks down with many options and long states, because options compete with the
state for context and positions. Temperature per (type x option count) is
needed. Confident failure on out-of-distribution input is the main risk.

### 2.3 Microsoft-Decision-1

- [P] A decision-scoring model post-trained from Qwen3.5-9B. 32,768-token
  context, text only, hosted only (Foundry, OpenRouter), no open weights.
- [P] Same request shape as JEV: `state`, `questions` map with `type`
  (noul/choice/score), `instructions`, `criteria`. Question keys are not shown
  to the model. Output is two tokens for an 86-token example.
- [V] 83.5% accuracy on 36 benchmarks (about 147k questions) vs Jev 82.3%.
  Calibration score 92.2 vs Jev 93.7 (100 is perfect; metric not specified).
- [V] Perturbation test: 1.3% answer changes; zero flips when options are
  paraphrased, reversed or shuffled.
- [V] 85 ms p50, 125 ms p95 in-region.
- [V] Supports explicit abstention ("cannot tell"). Out of scope as the sole
  basis for credit, employment, housing, insurance, education, healthcare, legal.

Design lessons: the strongest public system is a large pretrained LLM with its
decoder output restricted. Option-order robustness and explicit abstention are
table stakes. RHEN-2 gets order robustness structurally and makes abstention a
trained output.

### 2.4 How Claude reasons, and what transfers

From Anthropic's interpretability work [P] (studied on Claude 3.5 Haiku):

- **Refusal is the default.** A "cannot answer" circuit is on by default and is
  inhibited by "known entity" features. Hallucination happens when that
  inhibition misfires. -> RHEN-2 gives every question a null option whose score
  starts high. Real options must accumulate evidence to beat it.
- **Multi-step reasoning happens in latent space** (Dallas -> Texas -> Austin),
  and intervening on the intermediate concept changes the answer. -> Latent
  loops are a legitimate place to reason; causal intervention is the right way
  to test what a decision depends on.
- **Planning ahead** and **parallel pathways** (rough magnitude plus exact last
  digit for arithmetic). -> Give numbers a dedicated magnitude channel instead
  of hoping token embeddings encode quantity.
- **Stated reasoning can be unfaithful.** The model describes the school
  algorithm while computing differently, and works backward from hints. A
  separate study found hints that changed the answer were mentioned only 25% to
  39% of the time. -> A checker must not trust a self-report. RHEN-2's Checker
  tests decisions by intervention (remove the evidence, see if the answer
  moves), not by asking the model to explain itself.

From the extended-thinking product docs [P]:

- Thinking is a budget, now adaptive: an `effort` level decides whether and how
  much to think, and thinking is interleaved between actions. -> RHEN-2 exposes
  `effort` in {low, medium, high} that maps to a loop budget and to which
  Checker stages run. More compute is spent only when the Checker asks for it.

Important difference: Claude's reasoning is autoregressive text. RHEN-2's is
latent and bounded. RHEN-2 will not match an LLM on open-ended multi-hop
reasoning; the pipeline escalates those cases instead of pretending.

### 2.5 Looped and recurrent-depth models

- Huginn (3.5B, 800B tokens) [P]: prelude, a core iterated `r` times with the
  input embedding re-injected each step, then a coda. `r` sampled per sequence
  (log-normal Poisson, mean 32). Backprop only through the last 8 iterations.
  Without a "sandwich" normalisation the hidden states collapsed; one run learned
  to ignore the recurrent state entirely. Needed a low peak LR (4e-5).
- Tiny Recursive Model [P]: one tiny 2-layer network, a latent updated several
  times then the answer updated once; most recursions run without gradient;
  **deep supervision** (a loss at every outer step) is the largest single gain;
  EMA of weights matters; a cheap BCE halting head works.
- Loopie [P]: looping each layer twice, with compute matched by measured step
  time, beats a non-looped MoE model after enough tokens, and trails early.
- arXiv 2609.16665 (abstract only): extra iterations can overshoot; a quarter
  step fixes most such failures.
- MixerLoop (abstract only): looping only the mixer and applying the FFN once
  keeps much of the gain for fewer FLOPs.

Design lessons: input re-injection, sampled loop counts, truncated backprop,
deep supervision, strong normalisation, a damped update and EMA are the working
recipe. v1 has the damped update and little else on the decision path (A7).

### 2.6 Sources

- https://www.firecrawl.dev/blog/what-is-jev
- https://vizuara.substack.com/p/jev-what-a-model-built-for-decisions
- https://huggingface.co/blog/sora-2/jev-system-one-model-how-typed-ai-decisions-work-i
- https://anthonymaio.substack.com/p/jev-the-language-model-that-wont
- https://systemonemodels.org/guides/rlcd-explained/
- https://arxiv.org/abs/2609.28940
- https://www.eesel.ai/blog/laya-ai
- https://dev.to/vishalmysore/what-is-laya-laya-vs-jev-with-live-demo-4j6e
- https://purpleblob.net/en/blog/what-is-laya-decision-models/
- https://labs.ai.azure.com/innovations/microsoft-decision-1/
- https://techcommunity.microsoft.com/blog/educatordeveloperblog/not-every-decision-needs-an-llm-getting-started-with-microsoft-decision-1/4563468
- https://www.eesel.ai/blog/microsoft-decision-1
- https://www.anthropic.com/news/tracing-thoughts-language-model
- https://www.anthropic.com/research/reasoning-models-dont-say-think
- https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- https://arxiv.org/abs/2502.05171 (Huginn)
- https://arxiv.org/abs/2510.04871 (Tiny Recursive Model)
- https://arxiv.org/abs/2607.16051 (Loopie)
- https://arxiv.org/abs/2609.16665 and https://arxiv.org/abs/2608.18230 (abstracts only)

Not read, worth reading before Phase 2: TypeSafe's launch post and
`docs.typesafe.ai/model-jaggedness`, `evals.typesafe.ai`, the Laya model card on
Hugging Face, arXiv 2610.00673, 2610.26030 and 2610.23036 (looped-model training
recipes).

---

## 3. Deep-scan findings in the current code (v1)

Severity: **High** = the model does not compute what is documented, or training
cannot work well. **Med** = costs accuracy, speed or safety. **Low** = cleanup.
Line numbers are for commit `db66a9d`.

### 3.A Model

| ID | Sev | Where | Problem | Fix in RHEN-2 |
| :--- | :--- | :--- | :--- | :--- |
| A1 | High | `rhen/model.py:666` | Every odd layer adds a causal sliding window of 512 tokens on top of the branch mask (`interleave_swa` defaults on). In those layers the state is causal, not bidirectional, and a branch token more than 512 positions past a state token cannot see it. Later branches see less of the state than earlier ones. | State encoder has its own explicit local/global schedule; workspace reads the state by cross-attention with no window. |
| A2 | High | `rhen/model.py:3164` | RoPE positions run straight through the packed sequence, so the distance from a branch to the state depends on how many branches precede it. Reordering the questions changes the answers. "Isolated branches" is not order-invariant. | Each question and each option has its own position range starting at 0. |
| A3 | High | `rhen/model.py:1885`, `:1911` | The Rhen slot channel attends over **all** positions with no mask and adds a steering vector back to **all** positions, including the state, on every loop. Information flows between branches and from options into the state. The KV gate is also global. This contradicts README guarantees 2 and 3. | No global side channel. All cross-question paths are removed by construction and tested (T-207). |
| A4 | High | `rhen/model.py:529-566`, `:2800` (`use_rhen=False`) | LM pretraining uses causal attention with the Rhen engine off. Decisions use a bidirectional state with Rhen on. The pointer head, slots and steering weights get no pretraining signal, and the trunk never saw the attention pattern it is used with. | Pretraining objectives run through the same forward path as decisions (Section 7.2). |
| A5 | Med | `rhen/optim/rhen_mask.py:66` | Inside a branch, attention is causal, so option k sees only the options before it. Order invariance is only encouraged by a loss that needs a second full forward pass (`rhen/research/train.py:197`), doubling decision-training cost. | Option Set Layer: invariance holds exactly; the loss and second pass are deleted. |
| A6 | Med | `rhen/model.py:3080-3400`, `rhen/research/train.py:189` | `forward_rhen` indexes `[0]` everywhere and the trainer loops over examples in Python. Effective batch size is 1. | Fully batched padded tensors (Section 4.2). |
| A7 | Med | `rhen/model.py:3178` | On the decision path there is no input re-injection, no deep supervision, no halting, and full backprop through all 12 loops. `residual_scale = 1/sqrt(num_loops)` means changing the loop count at inference rescales every residual branch. ACT halting exists only on the LM path, so "adaptive depth" does not apply to decisions. | Section 4.5 recipe. |
| A8 | Med | `rhen/model.py:1750` | Slot state and a trajectory buffer are stored on the module (`_active_slots`). Not re-entrant; unsafe for concurrent serving. | All state is passed as function arguments and returned. |
| A9 | Med | `rhen/model.py:3199` | If `enable_rhen=False`, `forward_rhen` builds a new randomly initialised pointer head on every call and returns its output. Silent random decisions. | Raise an error. Also fix in v1 (task T-001). |
| A10 | Med | `rhen/rhen_loss.py:381`, `rhen/model.py:1642` | The Range head is trained with MSE on the mean only. The interval is `1.96 * sigma` of a Beta variance that never receives a training signal, and the "confidence" is `exp(-2 sigma)`. The intervals are not meaningful. | Quantile head trained with pinball loss (Section 4.7). |
| A11 | Low | `rhen/model.py:3263` | Extract picks the best start greedily, then the best end after it. No joint best span, no maximum length, no "not present" answer. | Joint span search with a null span. |
| A12 | Low | `rhen/model.py:1423` | The four pointer heads are documented as semantic, policy, severity and entity heads. Nothing in the code or losses makes them specialise. | Drop the names; one readout. |
| A13 | Low | `rhen/model.py:1311-1420` | `LatentTransitionOperator`, `ProspectiveValueHead` and `SchemaMutationHead` have no training objective in `rhen/rhen_loss.py` or `rhen/research/`. Their outputs at inference are from untrained weights. | Not carried over. |
| A14 | Med | `rhen/config.py` | Generation-only machinery in a decision model: MLA low-rank KV compression (helps only an autoregressive KV cache), multi-token prediction, speculative decoding, KV quantisation, YaRN, per-token ACT, Titans, SSM highway, CLAM, Hopfield memory, differential attention, 8-expert MoE at width 560. None has an ablation. `EOSConfig` has 128 keyword parameters. | RHEN-2 starts minimal. A component is added only after it wins an ablation at the 50M scale (Section 10.4). |
| A15 | Low | `rhen/config.py:357` | The embedding table is 50,048 x 560 = 28M parameters. At a 50M budget a full-width table would dominate. | Factorised embedding (Section 5). |

### 3.B Training and data

| ID | Sev | Where | Problem | Fix |
| :--- | :--- | :--- | :--- | :--- |
| B1 | High | `rhen/research/tasks.py` | The only decision training data is six synthetic template families. No converters from public datasets, no teacher distillation, no hard negatives. | Section 7. |
| B2 | Med | `rhen/rhen_cognition.py:103` | `RLCDOptimizer` is exported but never called by `rhen/research/train.py`. Training is cross-entropy plus Brier. The "RLCD" name in the README describes code that is not in the training loop. | Section 7.4 defines the calibrated objective explicitly; rename or wire it. |
| B3 | Med | `rhen/research/calibration.py` | One global temperature. Laya needed one per (question type x option count) to get from ECE 0.466 to 0.081. | Section 8.1. |
| B4 | Med | `rhen/research/train.py:173` | The pretraining objective is next-token prediction plus multi-token prediction. A decision model never generates. | Masked and span-choice pretraining (Section 7.2). |
| B5 | Low | `benchmarks/banking77/results.json` | The only RHEN figure is an untrained model on 3 examples. Honest, but there is no evaluation harness that runs the whole suite. | Section 10. |

### 3.C Pipelines and decision making

| ID | Sev | Where | Problem | Fix |
| :--- | :--- | :--- | :--- | :--- |
| C1 | Med | `rhen/rhen.py:1118`, `:2487`, `rhen/unified_orchestrator.py:36`, `rhen/rhen_pillars.py:172`, `rhen/decision_pipeline.py:423` | Five entry points (`RhenEngine`, `AdvancedRhenEngine`, `UnifiedAIOrchestrator`, `PillarStack`, `ForecastAwareDecider`) with different gating rules. The same decision can be executed by one and blocked by another. | One `DecisionPipeline` (Section 9); the others become thin wrappers. |
| C2 | High | `rhen/rhen.py:1162`, `rhen/meta_cognitive_monitor.py:313`, `:38` | Thresholds are constants: free energy 4.5, conformal fallback `1 - alpha` when uncalibrated, hull distances. The uncalibrated conformal set is flagged but still feeds the abstention verdict. | Thresholds only from a fitted calibration file; fail closed otherwise. |
| C3 | Med | `rhen/meta_cognitive_monitor.py:226` | Free energy is `-tau * logsumexp(scores/tau)`. It grows in magnitude with `log K`, so one threshold cannot serve 2 options and 77 options. | Use `E + tau * log K`, and fit the threshold per K bucket. |
| C4 | Med | `rhen/meta_cognitive_monitor.py:353` | "Epistemic uncertainty" is `1 - (top1 - top2)`. That is a function of the same softmax and is not epistemic. | Depth disagreement and the Checker's correctness head. |
| C5 | Med | `rhen/rhen.py:2150-2330` | Confidence means something different for each primitive: rescaled max probability, `2|p-0.5|`, ordinal dispersion, `exp(-2 sigma)`, `p_start * p_end`. A single `min_confidence` in the policy compares numbers that are not comparable. | One definition everywhere: calibrated probability that the decision is correct. |
| C6 | Med | `rhen/decision_pipeline.py:112` | The policy ranks options by `P(option) x utility` with non-negative utilities. There is no cost for a wrong action and no cost for abstaining, so it cannot trade risk against benefit. | Cost-matrix expected-loss rule with a reject option (Section 9.4). |
| C7 | Med | `rhen/meta_cognitive_monitor.py:25` | Split conformal with one global quantile gives marginal coverage only. Coverage can be poor inside a question type or an option-count bucket. | Group-conditional (Mondrian) conformal and conformal risk control per risk tier. |
| C8 | Low | `rhen/rhen_cognition.py:363-447` | `HierarchicalChoiceRouter` groups options by character-trigram Jaccard similarity. That is spelling similarity, not meaning. | RHEN-2 handles 255 options natively; above that, cluster on the model's own option vectors. |
| C9 | Low | `rhen/decision_pipeline.py:673` | The feedback ledger scores forecast calibration, not decision quality. | Record decision outcomes and feed online recalibration and drift alarms (Section 9.6). |
| C10 | Low | `rhen/meta_cognitive_monitor.py:628` | `decision_history` is an unbounded in-memory list. | Bounded ring buffer plus the ledger. |

### 3.D Repository

| ID | Sev | Problem | Fix |
| :--- | :--- | :--- | :--- |
| D1 | Med | Three parallel module trees are packaged together: root shims (`model.py`, ...), `rhen/`, and `optim/` plus `eos_research/`. Root shims replace themselves in `sys.modules`. Code inside `rhen/research/train.py` imports through the shims (`from rhen_packer import ...`). | RHEN-2 uses only absolute `rhen.v2.*` imports. Remove the shims in a later major version. |
| D2 | Low | The README shows three arXiv badges (2609.30266, 2609.29327, 2609.28879) in the header, where readers expect the project's own papers. They are third-party citations. | Move them to a References section. |
| D3 | Low | CI runs the full torch test suite on 12 OS/Python combinations. | Keep one full run and a smoke matrix. |

What is good and should be kept: the action-card signing, dual custody, Merkle
ledger and out-of-band audit code; the honest "Known limitations" section; the
delimiter sanitisation in the packer; the Banking77 harness with numpy
baselines; the checkpoint and distributed-training plumbing in `rhen/research/`.

---

## 4. RHEN-2 architecture: Cached-State Deliberation Lattice

### 4.1 Overview

```
state text ──► Tokenise ──► STATE ENCODER (bidirectional, run once) ──► state memory  S  [B, Ls, d]
                                                                              │ (cacheable by state hash)
questions ──► Tokenise ─┐                                                     │
options   ──► Tokenise ─┤                                                     ▼
                        └──► DELIBERATION WORKSPACE, loop r = 1..R  (shared weights)
                               1. question tokens:  self-attn + cross-attn to S
                               2. option tokens:    self-attn (own option only)
                                                    + cross-attn to [S ; question]
                               3. OPTION SET LAYER: pooled options + null node, no positions
                               4. readout at this depth (deep supervision), halting score
                                                     │
                                                     ▼
                               TYPED READOUTS: choice / noul / score / multichoice / range / extract
                                                     │
                                                     ▼
                               CHECKER: P(correct), evidence spans, interventions, depth consistency
                                        verdict = CONFIRM | REVISE (more loops) | ESCALATE
```

Why this is structurally different from v1, JEV-style encoders and LLMs:

| Property | Decoder LLM | Encoder + head (Laya) | v1 | RHEN-2 |
| :--- | :--- | :--- | :--- | :--- |
| State read | every token step | once, shared with options | once per call, re-processed in all loops | once, cached across calls |
| Options | generated or in prompt | in the same sequence | in the same sequence, causal | separate set, equal positions |
| Order invariance | no | no | by loss only | by construction |
| Question isolation | no | one question per pass | claimed, not true | by construction |
| Iterative reasoning | via tokens | none | loops over everything | loops over the small workspace only |
| Abstention | prompt | threshold | threshold | trained null option + Checker |
| Self-check | second prompt | none | heuristics | built-in interventional Checker |
| Cost of M questions | M generations | M passes | 1 long pass | 1 state pass + M small workspaces |

### 4.2 Input format and batching

File: `rhen/v2/packing.py`.

Reuse the existing tokenizer (`rhen/tokenizer.py`, vocabulary 50,048) and
`sanitize_delimiter_text` from `rhen/rhen_packer.py`. Add special tokens:
`<|null|>`, `<|type:choice|>`, `<|type:noul|>`, `<|type:score|>`,
`<|type:multi|>`, `<|type:range|>`, `<|type:extract|>`, `<|criteria|>`, `<|mask|>`.

A question follows the JEV and Microsoft request shape so that the same client
code can target any of the three systems:

```python
@dataclass
class QuestionSpec:
    type: str                 # "choice" | "noul" | "score" | "multi" | "range" | "extract"
    instructions: str         # the question text
    options: list[str]        # option labels; ["true","false"] is implied for noul
    criteria: list[str]       # optional description per option (same length as options) or []
    lo: float | None = None   # range only
    hi: float | None = None   # range only
```

Token layout (no state and no other question appears in these sequences):

- State: `<|state|> state_text`. Shape `[Ls]`.
- Question: `<|type:X|> instructions`. Shape `[Lq]`, `Lq <= 128`.
- Option k: `label <|criteria|> criteria_text`. Shape `[Lo]`, `Lo <= 48`.
  Question keys or ids are never tokenised (as in Microsoft-Decision-1).
- Score: the options are the ordered levels, lowest first.
- Noul: two fixed options, `true` and `false`. This makes
  `P(true) + P(false) + P(null) = 1` by construction, which removes the JEV
  inconsistency described in 2.1.
- Range and extract: no options.

Batch container (all padded, all on one device):

```python
@dataclass
class Rhen2Batch:
    state_ids:   LongTensor   # [B, Ls]
    state_mask:  BoolTensor   # [B, Ls]            True = real token
    state_num:   FloatTensor  # [B, Ls, 4]         numeric features, zeros when not a number (4.3)
    q_ids:       LongTensor   # [B, M, Lq]
    q_mask:      BoolTensor   # [B, M, Lq]
    q_type:      LongTensor   # [B, M]             0..5
    q_valid:     BoolTensor   # [B, M]             False = padded question
    opt_ids:     LongTensor   # [B, M, K, Lo]
    opt_mask:    BoolTensor   # [B, M, K, Lo]
    opt_valid:   BoolTensor   # [B, M, K]          False = padded option
    range_lo:    FloatTensor  # [B, M]
    range_hi:    FloatTensor  # [B, M]
```

Bucketing: the collator groups examples so that `K` and `Ls` are similar inside
a batch (bucket edges for K: 2, 4, 8, 16, 32, 64, 128, 255). This keeps padding
below about 20%.

Limits per scale are in Section 5. `K <= 255` everywhere.

### 4.3 Embeddings

File: `rhen/v2/layers.py`.

- **Factorised token embedding**: `Embedding(V, e)` then `Linear(e, d)`, with
  `e < d`. One table shared by state, question and option tokens.
- **Segment embedding**: 3 learned vectors of size `d` (state, question, option).
- **Type embedding**: 6 learned vectors added to every question token.
- **Positions**: rotary (reuse `RotaryEmbedding` from `rhen/model.py`, plain
  RoPE, no YaRN). Positions start at 0 separately for the state, for each
  question, and for each option. This fixes A2.
- **Numeric channel** (addresses the arithmetic and comparison weak spot): the
  packer detects numbers in the state and question with a regex
  (`[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?%?`). For the first token of each number
  it writes 4 features: `sign`, `log10(|x|+1)/12`, `frac(log10(|x|+1))`, and
  `is_percent`. A 2-layer MLP `4 -> d/4 -> d` maps them to a vector that is
  added to that token's embedding. All other tokens get zeros. Dates are
  converted to days since 2000-01-01 and treated as a number with a fifth
  flag folded into `is_percent`'s slot as value 2. Keep this simple; it is
  ablated in 10.4.

### 4.4 State Encoder

File: `rhen/v2/state_encoder.py`. Class `StateEncoder(cfg)`.

- `n_state_layers` pre-norm blocks: `x = x + Attn(RMSNorm(x))`,
  `x = x + SwiGLU(RMSNorm(x))`. Hidden size of SwiGLU is `int(8/3 * d)`.
  No biases. QK-norm on (RMSNorm on per-head queries and keys). Full-rank
  attention projections (no MLA low-rank factors).
- Attention is **bidirectional**. Layer schedule: every third layer is global
  (attends to all state tokens), the others are local with a symmetric window
  of 128 tokens (64 each side). At the 50M scale with `Ls <= 2048` all layers
  may be global; the schedule is a config field `global_every` (0 = all global).
- Padding handled by `state_mask`. Use
  `F.scaled_dot_product_attention(..., attn_mask=...)` so flash kernels apply.
- Output: `S` with shape `[B, Ls, d]`, after a final RMSNorm.
- The encoder never sees questions or options. Therefore `S` depends only on
  the state, and can be cached under a hash of `(model_version, state_ids)`.
  Cascades, retries, Checker interventions and follow-up questions on the same
  state cost no state compute.

### 4.5 Deliberation Workspace

File: `rhen/v2/workspace.py`. Class `DeliberationWorkspace(cfg)`.

One **core block** with weights shared across all loops and across the question
pass and the option pass:

```
CoreBlock(x, memory, memory_mask):
    x = x + SelfAttn(RMSNorm(x))                 # within the token group only
    x = x + CrossAttn(RMSNorm(x), memory)        # read memory
    x = x + SwiGLU(RMSNorm(x))
    return RMSNorm_out(x)                        # "sandwich" norm, required for loop stability
```

`n_core_layers` core blocks are stacked; the stack is applied `R` times.

Shapes. Flatten questions into the batch dimension: `N = B * M`.

- Question tokens `Q`: `[N, Lq + n_scratch, d]`. `n_scratch` learned scratch
  vectors are appended to every question (default 8). They hold intermediate
  results across loops; they are the workspace's "working memory".
- Option tokens `O`: `[N * K, Lo, d]`.
- State memory for question n: row `n // M` of `S`. Use `expand` and
  index, never copy.

One loop `r` (r = 1..R):

```
# 0. input re-injection (Huginn recipe): keep the loop anchored to its input
Q_in = Q + Q0 ;  O_in = O + O0            # Q0, O0 are the initial embeddings

# 1. question pass: memory = state
Q_new = CoreStack(Q_in, memory=S_n, memory_mask=state_mask_n)

# 2. option pass: memory = [state ; question tokens], with segment embeddings
mem   = concat(S_n, Q_new) repeated for the K options of that question
O_new = CoreStack(O_in, memory=mem)       # self-attn sees only this option's own Lo tokens

# 3. option set layer (4.6): mixes information across the K options, order-free
O_new = O_new + OptionSetLayer(pool(O_new), pool(Q_new), opt_valid)

# 4. damped update (fixes overshoot, arXiv 2609.16665)
alpha = sigmoid(step_logit)               # one learned scalar, init so alpha = 0.5
Q = Q + alpha * (Q_new - Q) ;  O = O + alpha * (O_new - O)

# 5. readout and halting at this depth
logits_r = Readout(Q, O)                  # 4.7
halt_r   = sigmoid(HaltHead(q_vec))       # [N], probability the decision is final
```

Isolation guarantees (each one gets a unit test in T-207):

- A question's tokens attend only to themselves and to `S`. No question sees
  another question. No parameter or buffer carries state between questions.
- An option's tokens attend only to their own option, to `S` and to their
  question. Options interact only in the Option Set Layer.
- Nothing writes into `S`.

Loop schedule:

- Training: sample `R` once per batch from a truncated distribution with mean
  `r_mean` in `[r_min, r_max]` (use `R = clamp(Poisson(r_mean - r_min) + r_min,
  r_min, r_max)`). Run the first `R - k_bptt` loops under `torch.no_grad()` and
  the last `k_bptt` loops with gradient (`k_bptt = 2`). This is the truncated
  backprop recipe from Huginn and TRM; memory cost is independent of `R`.
- Deep supervision: the task loss is applied to `logits_r` for each of the
  gradient-carrying loops, weighted `0.5` for the earlier and `1.0` for the last.
- Halting target for loop r: `1` if `argmax(logits_r)` equals the gold label,
  else `0` (TRM-style BCE). Weight 0.1.
- Inference: run at least `r_min` loops; stop when `halt_r > halt_threshold`
  (from the calibration file, default 0.9) or at the effort's loop cap
  (Section 6.5). Because questions are independent, halting is per question:
  finished questions are dropped from the batch.
- Keep an exponential moving average of the weights (decay 0.999) and evaluate
  with the EMA weights.

Large option sets: when `K * Lo * Ls` exceeds `evidence_budget` (default 4M
attention pairs), options cross-attend to an **evidence subset** of the state
instead of all of it: the top `n_evidence` (default 256) state positions by the
question pass's cross-attention mass in the current loop, always including
position 0. The question pass still reads the full state. This keeps 255-option
questions on a 32k state affordable.

### 4.6 Option Set Layer

File: `rhen/v2/workspace.py`. Class `OptionSetLayer(cfg)`.

- Pool each option to one vector with learned attention pooling:
  `u_k = sum_t softmax(w^T O[k,t]) * O[k,t]`, shape `[N, K, d]`.
- Pool the question the same way to `q_vec`, shape `[N, d]`.
- Build the set `[q_vec, null_vec, u_1 .. u_K]`, where `null_vec` is a learned
  parameter (the "cannot tell" node). **No positional information of any kind**
  is added to set members.
- Apply `n_set_layers` standard pre-norm self-attention blocks over the set
  with `opt_valid` as padding mask.
- Return the updated option vectors `u'_k`; the workspace adds `u'_k - u_k` to
  every token of option k (broadcast over `Lo`). The updated `null_vec'` and
  `q_vec'` are kept for the readout.

Self-attention without positions is permutation-equivariant, and the readout
scores each option with shared weights, so permuting the options permutes the
output probabilities exactly. Test T-206 checks this to `1e-5`. The
`compute_option_order_consistency_loss` second forward pass is not needed.

For Score questions only, add a learned **rank embedding** `rank_emb[k]` to
`u_k` before the set layer, because levels are ordered.

### 4.7 Typed readouts

File: `rhen/v2/readouts.py`. All readouts consume `q_vec'`, `u'_k`, `null_vec'`.

Base score for option k:
`s_k = <RMSNorm(W_q q_vec'), RMSNorm(W_o u'_k)> * exp(log_scale)`, with
`W_q, W_o: d -> d_r` (`d_r = 256`), and a learned `log_scale` (init `log 10`).
The null node gets `s_null` the same way plus a learned bias initialised to
`+2.0`, so "cannot tell" is the default until evidence accumulates (2.4).

| Type | Output | Formula | Loss |
| :--- | :--- | :--- | :--- |
| choice | `P` over K options + null | `softmax([s_1..s_K, s_null] / T)` | log loss + 0.5 x Brier |
| noul | `P(true), P(false), P(null)` | same as choice with K = 2 | same |
| score | distribution over ordered levels + null | cumulative-link: `P(level <= j) = sigmoid(theta_j - g)`, `g = MLP(q_vec')`, thresholds `theta` strictly increasing via cumulative softplus, one threshold set generated from the level vectors `u'_k`; null handled by a separate sigmoid gate | ranked probability score + 0.5 x log loss |
| multi | independent `P_k` | `sigmoid((s_k - s_null) / T)` | per-option BCE + 0.5 x Brier |
| range | 9 quantiles (0.1 .. 0.9) in `[lo, hi]` | `lo + (hi-lo) * cumsum(softmax(MLP(q_vec')))` gives monotone quantiles | pinball loss averaged over the 9 levels |
| extract | start and end over state tokens + null span | `start = <W_s q_vec', S>`, `end = <W_e q_vec', S>`, null span = position 0; answer = argmax over `start_i + end_j` with `i <= j <= i + max_span` | cross-entropy on start and end |

`T` is a per-question temperature. In the model it is fixed at 1; calibration
(Section 8.1) supplies the value at inference. Do not learn an input-dependent
temperature inside the model: v1 tried (`entropy_proj`) and had to patch it.

The quantile format for Range matches `Forecast` in `rhen/forecasting.py`
(nine quantiles, 0.1 to 0.9), so the existing interval and exceedance code can
be reused.

### 4.8 Model class and forward signature

File: `rhen/v2/model.py`.

```python
class Rhen2Model(nn.Module):
    def encode_state(self, state_ids, state_mask, state_num) -> Tensor:        # [B, Ls, d]
    def deliberate(self, S, state_mask, batch: Rhen2Batch, *, loops: LoopPlan,
                   interventions: Interventions | None = None) -> Rhen2Output:
    def forward(self, batch, *, loops, labels=None) -> Rhen2Output:            # encode + deliberate + losses

@dataclass
class Rhen2Output:
    logits:      Tensor        # [N, K+1]  last loop, null is the last column
    per_loop:    list[Tensor]  # logits for each gradient-carrying loop
    halt:        Tensor        # [N, R_run]
    quantiles:   Tensor        # [N, 9]    range questions, else zeros
    span_start:  Tensor        # [N, Ls]
    span_end:    Tensor        # [N, Ls]
    evidence:    Tensor        # [N, Ls]   state attribution (6.2)
    p_correct:   Tensor        # [N]       Checker head (6.1)
    loops_run:   LongTensor    # [N]
    loss:        Tensor | None
    loss_parts:  dict[str, float]
```

The model holds no per-call state. `encode_state` and `deliberate` are separate
so the pipeline can cache `S`.

---

## 5. Scale ladder

Parameter formulas (no biases): attention `4 d^2`; SwiGLU `3 * d * int(8d/3)`
which is about `8 d^2`; encoder layer `12 d^2`; core layer (self + cross + FFN)
`16 d^2`; set layer `12 d^2`; embedding `V*e + e*d`.

| | **R2-50M** | **R2-300M** | **R2-1B** | **R2-8B** |
| :--- | :--- | :--- | :--- | :--- |
| width `d` | 512 | 1024 | 1536 | 3072 |
| embedding dim `e` | 192 | 384 | 768 | 1536 |
| heads (head dim 64) | 8 | 16 | 24 | 48 |
| state encoder layers | 8 | 16 | 24 | 54 |
| core layers (shared) | 3 | 4 | 6 | 10 |
| set layers | 1 | 2 | 2 | 2 |
| loops `r_min / r_mean / r_max` | 2 / 4 / 8 | 2 / 4 / 8 | 2 / 4 / 12 | 2 / 4 / 16 |
| scratch slots | 8 | 8 | 16 | 32 |
| max state tokens `Ls` | 2,048 | 8,192 | 16,384 | 32,768 |
| `global_every` | 0 (all global) | 3 | 3 | 3 |
| evidence subset | off | 256 | 384 | 512 |
| embedding params | 9.7M | 19.6M | 39.6M | 81.6M |
| state encoder params | 25.2M | 201.3M | 679.5M | 6,115M |
| core params | 12.6M | 67.1M | 226.5M | 1,510M |
| set layer params | 3.1M | 25.2M | 56.6M | 226.5M |
| heads + MLM transform | 1.8M | 5.1M | 10.2M | 34.6M |
| **total** | **52.4M** | **318.3M** | **1,012M** | **7,968M** |
| effective depth at `r_mean` | 8 + 12 = 20 | 16 + 16 = 32 | 24 + 24 = 48 | 54 + 40 = 94 |

To land R2-50M at or under 50.0M, set `e = 160` (saves 1.6M) and
`n_scratch = 8`; either is acceptable. Record the exact count from
`sum(p.numel() ...)` in the run manifest.

File: `rhen/v2/config.py` holds one `Rhen2Config` dataclass (target: under 40
fields) and a `PRESETS` dict with these four entries plus `r2-tiny`
(`d=64, e=32, 2 encoder layers, 1 core layer, 1 set layer`) for tests.

### 5.1 What changes at each step up

The module code is identical at all scales. Only config values and the
training/serving setup change. Do the steps in order; each step lists exactly
what to modify.

**50M (first target).**
- Single GPU (24 GB is enough), bf16 autocast, AdamW.
- All-global attention; no evidence subset; no gradient checkpointing.
- Mixed decision batches of about 64 questions.
- Purpose: prove the architecture, run every ablation in 10.4 here because it
  is cheap. Do not start 300M until the 50M gates (10.3) pass.

**50M -> 300M.** Modify:
1. Preset values as in the table.
2. Turn on the local/global schedule (`global_every = 3`, window 128).
3. Turn on the evidence subset (256).
4. Scale hyperparameters with width using the rule already in
   `rhen/optim/mup.py`: hidden-matrix learning rate proportional to `1/d`
   relative to the 50M base, embedding and readout learning rates unchanged,
   init std `1/sqrt(d)`. Re-tune only the peak LR on a 3-point sweep.
5. Multi-GPU with DDP (reuse `rhen/research/distributed.py`).
6. State sequence length curriculum: 2k for the first 60% of pretraining, then
   8k.

**300M -> 1B.** Modify:
1. Preset values.
2. Gradient checkpointing on the state encoder (reuse
   `rhen/research/checkpointing.py` helpers); FSDP.
3. Tokenizer: retrain to a 64k multilingual vocabulary if non-English is a
   goal (Laya's English checkpoint scored 0.000 on Khmer). This changes `V`;
   do it before 1B pretraining, never between pretraining and fine-tuning.
4. State cache in serving becomes mandatory (16k states).
5. The 1B model becomes the **teacher**: re-distil the 50M and 300M students
   from its soft labels (Section 7.5). Expect the distilled 50M to beat the
   directly trained 50M.

**1B -> 8B.** Modify:
1. Preset values. Two honest options for the backbone:
   - (a) Train the 8B CSDL from scratch. Needs on the order of a trillion
     pretraining tokens (Huginn used 800B tokens at 3.5B parameters). Only
     choose this with a compute budget in the range of thousands of
     high-end-GPU days. This is an estimate, not a measurement.
   - (b) Recommended: use a pretrained open LLM of similar size as the **State
     Encoder** through an adapter (this is what Microsoft did with Qwen3.5-9B),
     convert its attention to bidirectional with a short masked-token
     adaptation phase, and train the workspace, set layer, readouts and Checker
     from scratch on top. v1's `rhen/adapter.py` is the starting point.
2. Optional MoE feed-forward in the upper half of the state encoder (8 experts,
   top-2), reusing `SharedExpertFineGrainedMoE` from `rhen/model.py`. Add it
   only if a 1B-scale ablation shows a gain at matched step time.
3. Serving: tensor parallel or 8-bit weights; state cache; batch the workspace
   across requests that share a state.
4. Distil 8B -> 1B -> 300M -> 50M for the production ladder.

### 5.2 Latency design targets (not measured)

| Scale | Hardware | State 1k tokens, 1 question x 8 options, medium effort |
| :--- | :--- | :--- |
| 50M | CPU 8 cores | under 60 ms |
| 50M | T4-class GPU | under 10 ms |
| 300M | T4-class GPU | under 40 ms (Laya: 33 to 40 ms) |
| 1B | A10-class GPU | under 80 ms (Microsoft: 85 ms p50) |
| 8B | A100-class GPU | under 250 ms (JEV third-party: 236 to 276 ms) |

With a warm state cache, an additional question on the same state should cost
under 30% of the cold figure. These numbers must be replaced by measured ones
from `benchmarks/latency/` before any is quoted in the README.

---

## 6. The built-in reasoner: the Checker

File: `rhen/v2/checker.py`. The Checker is part of the model package and runs
inside `DecisionPipeline`. It never trusts a self-report; it measures.

### 6.1 Stage C1: probability of correct (always on)

A small head predicts whether the decision is right.

Input features, concatenated:
- `q_vec'` and the chosen option vector `u'_chosen` (each projected to 128)
- top-1 probability, top-1 minus top-2 margin, `P(null)`, entropy / log(K+1)
- depth features: fraction of executed loops whose argmax equals the final
  argmax; L2 distance between the last two loops' `q_vec`
- evidence concentration: entropy of the evidence distribution (6.2)
- `log K`, question type one-hot, `log Ls`

Head: `MLP(in -> 256 -> 64 -> 1)`, sigmoid. Output `p_correct`, shape `[N]`.

Training (Section 7.6): cross-fitted. Never train C1 on predictions the trunk
made on its own training data, or it will learn that the model is always right.

`p_correct` after isotonic calibration **is** the single confidence number the
pipeline uses for every question type. This fixes C5.

### 6.2 Stage C2: evidence (always on)

For each question, the evidence distribution over state tokens is the
cross-attention of the question tokens to the state in the last loop, averaged
over heads and over question tokens, then multiplied by the gradient-free
"attention rollout" through the core layers. Shape `[N, Ls]`, sums to 1.

Return the top spans (merge adjacent tokens above `1/Ls * 4`, at most 5 spans)
as `evidence_spans: list[(start_char, end_char, weight)]`.

Attention is not automatically a faithful explanation. C3a tests it.

### 6.3 Stage C3: interventions (medium effort and above, or risk tier >= high)

Each intervention re-runs only the workspace on a modified memory. The state
encoder is not re-run unless stated.

- **C3a evidence deletion.** Replace the memory vectors of the top evidence
  spans with a learned `mask_memory` vector and re-run. Expectation: the chosen
  option's probability drops by at least `delta_ev` (calibrated). If it does
  not, the decision does not depend on the cited evidence: flag `UNGROUNDED`.
  For risk tier `critical`, also do the exact version: mask the tokens in the
  input and re-run the state encoder.
- **C3b state removal.** Re-run with an empty memory (only position 0).
  Expectation: the model moves towards the null option. If it gives the same
  answer with similar confidence, it is answering from prior knowledge of the
  question wording, not from the state: flag `PRIOR_DRIVEN`.
- **C3c depth consistency.** Run to the effort's maximum loops. If the argmax at
  the halting loop differs from the argmax at the maximum loop, flag `UNSTABLE`.
- **C3d option perturbation.** Re-run with the null bias raised by 1.0. If the
  answer flips to null, the margin over "cannot tell" was thin: flag `MARGINAL`.

All four reuse the cached `S`, so a full C3 costs about four workspace passes.

### 6.4 Stage C4: deterministic checks (always on, cheap)

Reuse and extend `CheapVerifier` in `rhen/rhen_verify.py`:
- If the question is a numeric or date comparison and both operands can be
  parsed from the evidence spans, compute the answer exactly. On disagreement
  with the model, flag `RULE_CONFLICT` and prefer the deterministic answer.
- Extract: the returned string must be an exact substring of the state.
- Schema: the chosen label must be one of the supplied options (always true by
  construction; keep the assert).

### 6.5 Verdict and effort

```
effort   loops cap        stages run
low      r_min .. 2       C1, C2, C4
medium   halting, cap 4   C1, C2, C4, and C3 only if p_correct < tau_review
high     halting, cap max C1, C2, C3 (all), C4
```

Default effort is `medium`. Risk tier `high` or `critical` forces `high`.

Verdict rule, applied in this order:

1. `RULE_CONFLICT` -> return the deterministic answer, verdict `CONFIRM`,
   note the override.
2. top-1 is null, or the conformal set (8.2) is empty -> `ESCALATE` (abstain).
3. Any of `UNGROUNDED`, `PRIOR_DRIVEN` -> `ESCALATE`.
4. `UNSTABLE` or `MARGINAL`, and effort budget not exhausted -> `REVISE`: raise
   the loop cap to the next effort level and re-run once. If still flagged ->
   `ESCALATE`.
5. `p_correct >= tau_act[tier]` -> `CONFIRM`.
6. Otherwise -> `ESCALATE`.

`ESCALATE` means: do not act; hand to a human or to a System-2 LLM with the
state, the question, the probabilities and the evidence spans. This mirrors how
JEV, Laya and Microsoft all recommend their models be used.

Result object returned to callers:

```python
@dataclass
class CheckedDecision:
    answer: Any                      # label | bool | float | list[str] | (lo, hi) | str
    probabilities: dict[str, float]  # includes "__null__"
    confidence: float                # calibrated p_correct
    conformal_set: list[str]
    verdict: str                     # CONFIRM | ESCALATE
    flags: list[str]                 # UNGROUNDED, PRIOR_DRIVEN, UNSTABLE, MARGINAL, RULE_CONFLICT, OOD
    evidence_spans: list[tuple[int, int, float]]
    loops_run: int
    effort: str
    calibrated: bool
```

---

## 7. Data and training

The user will train later. This section is still part of the plan because the
architecture is only as good as this recipe, and the data code must exist
before training can start.

### 7.1 Data sources

Files under `rhen/v2/data/`.

1. **Text for pretraining** (`corpus.py`): general English text plus
   structured operational text (JSON, logs, tables, tickets, emails, configs),
   because decision states look like that. Target mix: 50% prose, 50%
   structured. Reuse `prepare_token_corpus` from `rhen/research/data.py`.
2. **Converted public datasets** (`converters/`): one small function per
   dataset that yields `(state, QuestionSpec, label)`. Start with:
   - noul: BoolQ, MNLI/SNLI/ANLI (entailment as a proposition), FEVER, QNLI
   - choice: AG News, DBpedia, TREC, Banking77, CLINC150 (includes out-of-scope
     as null), MASSIVE intents, RACE, CommonsenseQA, OpenBookQA, ARC
   - score: Yelp and Amazon star ratings, STS-B (binned), essay scoring sets
   - multi: GoEmotions, Reuters-21578 topics, toxic-comment labels
   - extract: SQuAD 2.0 (unanswerable -> null span), CoNLL NER as slot questions
   - range: synthetic from tables (7.1.3) and any regression-with-context sets
   Check each dataset's licence before use and record it in the manifest.
3. **Synthetic programmatic tasks** (`synth.py`): extend
   `rhen/research/tasks.py` from 6 to at least 40 families aimed at the known
   weak spots: number and date comparison, counting, thresholds, multi-hop
   lookups across a JSON state, policy rules with exceptions, large noisy
   states with one relevant line, adversarial text inside the state that tells
   the model which option to pick (the label ignores it).
4. **Teacher-labelled synthetic decisions** (`teacher.py`): an LLM writes
   states and questions for many domains; an ensemble of two or more strong
   LLMs returns a probability for each option; the average distribution is the
   soft label. Disagreement between teachers is kept as genuine uncertainty.
   This is the largest and most important source. Target sizes: 2M questions
   for 50M, 10M for 300M, 30M for 1B, 100M for 8B.
5. **Unanswerable examples**: 12% of all decision examples have the gold answer
   removed from the state or from the options; their label is null.

**Benchmark firewall.** The test split of every benchmark in Section 10.1 and
all of JEV's and Microsoft's published benchmark names go in
`rhen/v2/data/blocklist.json`. The data loader hashes each example and refuses
blocklisted ones. A test (T-304) enforces this.

### 7.2 Stage 1: pretraining (self-supervised, whole model)

Unlike v1, every parameter is trained by the same forward path used for
decisions. Three objectives, mixed per batch 40/40/20:

1. **Masked token modelling** on the state encoder: mask 30% of state tokens,
   predict them with an MLM transform (`Linear(d,d)`, GELU, RMSNorm, then the
   tied embedding matrix).
2. **Span-choice pretraining (SCP)**: take a passage, cut out a span of 3 to 30
   tokens, replace it with `<|mask|>`. The question is `<|type:choice|> which
   text fills the gap?`. The options are the true span plus `K-1` distractor
   spans (half sampled from the same document, half from other documents in
   the batch). K is sampled from {2, 4, 8, 16, 32}. With probability 0.12 the
   true span is left out and the label is null. This trains the workspace, the
   set layer, the readout and the null option with no labels.
3. **Span-locate pretraining**: copy a span from the passage into the question
   (`<|type:extract|> locate: <span>`), with 15% of copies lightly paraphrased
   by token dropout; the label is its position. Trains the extract readout.

Token budgets (rough guidance, to be revised from loss curves): 50M: 5 to 10B
tokens; 300M: 30 to 50B; 1B: 100 to 200B; 8B: 800B or more (or option (b) in 5.1).

### 7.3 Stage 2: supervised decision training

Mix: 45% teacher-labelled (7.1.4), 35% converted public datasets, 20%
programmatic synthetic. Keep 10% SCP in the mix to prevent forgetting.

Augmentations applied on the fly:
- shuffle option order (free, the model is invariant; this only checks data code)
- paraphrase the instructions (pre-generated 3 paraphrases per question)
- drop or add criteria text
- insert distractor paragraphs into the state
- add a `none of these` style option, or rely on the null node (50/50)
- option count augmentation: add irrelevant options sampled from other examples

### 7.4 Loss (the calibrated objective)

File: `rhen/v2/losses.py`.

```
L = L_task                      # proper scoring rule per type, table in 4.7
  + 0.5 * L_task(previous loop) # deep supervision
  + 0.1 * L_halt                # BCE, target = argmax correct at that loop
  + 0.1 * L_evidence            # only when rationale spans exist (SQuAD, FEVER): CE between evidence and span mask
  + 0.05 * L_null_margin        # hinge: on answerable examples, gold score exceeds null score by 1.0
```

With soft teacher labels, `L_task` is the cross-entropy to the soft
distribution plus 0.5 x Brier against it.

Every `L_task` is a strictly proper scoring rule, so the minimiser is the true
conditional probability. That is the published essence of "RLCD". No policy
gradient is needed for this part. The reinforcement part comes in Stage 4.

Optimiser: AdamW, betas (0.9, 0.95), weight decay 0.05 on matrices only,
gradient clip 1.0, warmup 2% of steps, cosine decay to 10% of peak. Peak LR
3e-4 at `d=512`, scaled as in 5.1. Looped models are LR-sensitive (Huginn
needed a 10x lower LR than its failed runs), so run a 3-point LR sweep at each
scale before the long run.

### 7.5 Stage 3: distillation down the ladder

Once a larger RHEN-2 exists, label the full training mix with it (soft
distributions at temperature 2) and train the smaller model on a 50/50 mix of
teacher distributions and original labels. Also distil the `p_correct` target.

### 7.6 Stage 4: calibration-aware fine-tuning and Checker training

1. Split the decision data into 5 folds. For each fold, take the Stage-2 model
   fine-tuned on the other 4 folds (a short fine-tune is enough) and predict
   the held-out fold. Collect `(features, was_correct)` pairs.
2. Train the C1 head on those pairs with BCE. Trunk frozen.
3. **Selective-decision fine-tuning (the "RL" in RLCD, our reading of it).**
   Treat act-vs-abstain as an action. Reward: `+1` for a correct act, `-c` for a
   wrong act (`c` sampled per batch from {1, 3, 9, 19}, corresponding to
   required accuracy 50%, 75%, 90%, 95%), `0` for abstain. The policy is
   `act if p_correct > c / (1 + c)`. Optimise the expected reward through a
   sigmoid relaxation of the threshold (temperature 0.1), updating the C1 head
   and the last core layer only, for a small number of steps. Stop when
   held-out area under the risk-coverage curve stops improving. The existing
   `RLCDOptimizer` in `rhen/rhen_cognition.py` should be adapted to this or
   renamed; do not leave a class named RLCD that is not used in training.
4. Train `mask_memory` and set `delta_ev` (6.3) on the rationale datasets.

---

## 8. Calibration layer (post-hoc, required before any deployment)

File: `rhen/v2/calibrate.py`. Output: one JSON file, `calibration.json`, stored
next to the checkpoint and hashed into the manifest.

### 8.1 Temperatures

Fit one temperature per group, where group = (question type, K bucket), K
buckets as in 4.2. Minimise NLL on a calibration split of at least 500 examples
per group; groups with fewer fall back to the type-level temperature. Extend
`fit_temperature` in `rhen/research/calibration.py`. Then fit isotonic
regression for `p_correct` per question type.

### 8.2 Conformal sets

Group-conditional split conformal: one quantile per (type, K bucket), score
`1 - p(true label)` after temperature. Report `calibrated: false` and an empty
guarantee if the group has fewer than 100 calibration points. For risk tiers,
use conformal risk control: choose `tau_act[tier]` as the smallest threshold
whose upper confidence bound on the error rate among acted decisions is below
the tier's tolerance (defaults: low 10%, medium 5%, high 1%, critical 0.1%).
Reuse `ConformalPredictionEngine` and `AdaptiveConformalEngine`.

### 8.3 OOD score

Three signals, each with a threshold fitted at the 99th percentile of
in-distribution calibration data per group:
- normalised energy `E + T * log(K + 1)` (fixes C3)
- `P(null)`
- Mahalanobis distance of `q_vec'` to the calibration-set mean (shared covariance)
Any signal over its threshold adds the `OOD` flag, which forces `ESCALATE`
above the `low` tier. The convex-hull gate in v1 (`GeometricAbstentionGate`)
may be kept as a fourth signal only if it improves OOD AUROC in 10.2.

### 8.4 Contents of `calibration.json`

`model_hash`, `data_hash`, `created`, `temperature[type][k_bucket]`,
`isotonic[type]`, `conformal_q[type][k_bucket]`, `tau_act[tier]`,
`tau_review`, `halt_threshold`, `delta_ev`, `ood[signal][type][k_bucket]`,
`n_calibration[type][k_bucket]`. The pipeline refuses a file whose `model_hash`
does not match the loaded checkpoint.

---

## 9. Unified decision pipeline

File: `rhen/v2/pipeline.py`. Class `DecisionPipeline`. This replaces the five
entry points in finding C1. After Phase 6, `RhenEngine`, `UnifiedAIOrchestrator`
and `ForecastAwareDecider` call it and keep their public signatures.

### 9.1 Stages, in fixed order

```
1  INGEST      sanitise delimiters, serialise state, detect numbers and dates, hash the state
2  PLAN        validate QuestionSpecs; split K > 255 with the hierarchical router; pick effort from risk tier
3  ENCODE      state memory S from cache or StateEncoder
4  DELIBERATE  workspace loops with per-question halting
5  CALIBRATE   temperatures, isotonic, conformal sets, OOD flags          (needs calibration.json)
6  CHECK       Checker stages by effort; REVISE loops back to 4 at most once
7  POLICY      expected-loss rule with constraints and risk tier -> EXECUTE | NEEDS_APPROVAL | ABSTAIN | BLOCKED
8  STAGE       ActionCard, signatures, dual custody, Merkle ledger         (existing rhen/agent_tools.py)
9  AUDIT       out-of-band record of inputs hash, outputs, flags, versions (existing rhen/rhen_verify.py)
10 FEEDBACK    outcome recording, online recalibration, drift alarms
```

Every stage appends a typed record to a `DecisionTrace`. The trace is what gets
audited and what a human reviewer sees.

### 9.2 Fail-closed rules

- No `calibration.json`, or hash mismatch: `calibrated = false`; nothing above
  the `low` tier may be `EXECUTE`.
- Any exception in stages 3 to 6: outcome `ABSTAIN` with the error recorded.
  Never fall back to a different scorer silently.
- A model with untrained weights (manifest has no `training_run_id`) may be
  loaded for tests only; the pipeline refuses to stage actions with it.

### 9.3 Question routing for very large option sets

For K > 255: get option vectors from one option pass with an empty question,
cluster them with k-means into at most 255 groups, ask the model to choose a
group (options are the group's three most central labels joined), then choose
inside the group. Replace the trigram clustering in `HierarchicalChoiceRouter`
(C8). Report top-1 accuracy against a flat split-and-merge baseline in 10.2
before enabling by default.

### 9.4 Policy: expected loss with a reject option

Replace `P(option) x utility` (C6). The caller supplies, per question, either a
full cost matrix `C[a][y]` (cost of taking action `a` when the truth is `y`) or
the shorthand `(cost_wrong, cost_abstain)` with `cost_right = 0`.

```
risk(a)      = sum_y P(y) * C[a][y]            # P is the calibrated distribution, null included
risk(abstain)= cost_abstain
a*           = argmin over admissible a and abstain
```

Constraints (existing predicate mechanism) remove inadmissible actions first.
Then:
- no admissible action -> `BLOCKED`
- `a*` is abstain, or the Checker verdict is `ESCALATE` -> `ABSTAIN`
- risk tier of `a*` requires approval -> `NEEDS_APPROVAL`
- otherwise -> `EXECUTE`

With the shorthand costs this reduces to "act if `p_correct > 1 -
cost_abstain / cost_wrong`", which ties the policy threshold to stated costs
instead of a magic number. Default costs per tier come from the tolerances in
8.2.

Multi-question workflows: calibrated parts do not compose into a calibrated
whole. For a cascade, the pipeline reports the product of the stage
confidences as a lower-bound estimate and labels it `uncalibrated_joint`,
unless a calibration set of whole workflows was provided.

### 9.5 Forecast-aware decisions

Keep `rhen/forecasting.py` and the breach-bound logic. Change only the
connection: forecast summaries are appended to the state before stage 1, the
forecast uncertainty enters the cost matrix (a wider interval raises
`cost_wrong` through a caller-supplied function) instead of being a separate
hard threshold.

### 9.6 Feedback and drift

- `FeedbackLedger.record_outcome` gains a `decision_correct: bool | None`.
- Every 500 labelled outcomes per group, refit that group's temperature and
  conformal quantile on a sliding window (keep the offline file as the floor:
  online updates may tighten, and may loosen only up to 2x the offline
  threshold before raising a drift alarm).
- Drift alarms: rolling ECE above 2x the offline ECE; OOD flag rate above 3x
  the calibration rate; adaptive conformal miss streak (existing logic).
  An alarm downgrades the affected group to `calibrated = false`.

---

## 10. Evaluation and gates

File: `rhen/v2/eval.py` and `benchmarks/suite/`. Every run writes a JSON file
with the dataset revision, hashes, commit, seed, environment and results, in
the format of `benchmarks/banking77/results.json`.

### 10.1 Benchmark suite

Accuracy and calibration (zero-shot and after per-task fine-tuning, reported
separately): Banking77, CLINC150 (with out-of-scope), SST-2, AG News, TREC,
MNLI, BoolQ, ARC, RACE, CommonsenseQA, GoEmotions, SQuAD 2.0, STS-B, Yelp
stars, the repo's synthetic `test` and `test_ood` splits, and a held-out
teacher-labelled set of 20k questions over 40 domains.

Published reference points to compare against (theirs, not reproduced here):
JEV: Banking77 0.870, CLINC150 0.895, SST-2 0.964, ARC 0.988, pooled ECE 0.028.
Laya: Banking77 0.425, ECE 0.081, about 33 to 40 ms on a T4.
Microsoft-Decision-1: 83.5% on its 36-benchmark suite, 85 ms p50.
If API access to JEV or Microsoft-Decision-1 is available, run them on the same
suite files and commit the outputs; only then may the README rank them.

### 10.2 Metrics

- top-1 accuracy with Wilson 95% interval
- ECE (15 bins, equal mass), Brier, NLL, before and after temperature
- selective prediction: area under the risk-coverage curve; accuracy at 50%,
  80%, 95% coverage; coverage at 1% error
- conformal: empirical coverage and mean set size per group at alpha 0.05, 0.1
- OOD: AUROC of each signal on `test_ood` and on a cross-dataset set
- robustness: flip rate under option shuffle (must be 0), question reorder
  (must be 0), instruction paraphrase, distractor injection, adversarial
  instruction inside the state
- Noul consistency: mean |P(p) + P(not p) - 1| on a negated-pairs set
- Checker: AUROC of `p_correct`; rate of `UNGROUNDED` and `PRIOR_DRIVEN` flags
  on correct vs wrong decisions; evidence deletion effect size
- weak-spot suite: arithmetic, counting, date comparison, multi-hop
- latency p50/p95 cold and warm cache, by state length and K; throughput

### 10.3 Gates (a scale may not start before the previous scale passes)

| Gate | 50M | 300M | 1B | 8B |
| :--- | :--- | :--- | :--- | :--- |
| Shuffle and reorder flip rate | 0 | 0 | 0 | 0 |
| Banking77, fine-tuned | >= 0.88 | >= 0.92 | >= 0.93 | >= 0.94 |
| Banking77, zero-shot | >= 0.45 | >= 0.70 | >= 0.85 | >= 0.88 |
| Suite mean accuracy, zero-shot | >= 0.55 | >= 0.68 | >= 0.78 | >= 0.83 |
| Pooled ECE after calibration | <= 0.06 | <= 0.04 | <= 0.03 | <= 0.03 |
| Conformal coverage at alpha 0.05, every group | 0.93 to 0.97 | same | same | same |
| OOD AUROC (`test_ood`) | >= 0.85 | >= 0.90 | >= 0.92 | >= 0.94 |
| `p_correct` AUROC | >= 0.80 | >= 0.85 | >= 0.88 | >= 0.90 |
| Latency | table 5.2 | table 5.2 | table 5.2 | table 5.2 |

"Comparable to JEV" is declared only when the 1B or 8B model is within 2
points of JEV's published accuracy on the shared public benchmarks and its
pooled ECE is at or below 0.03, with the result files committed. These gate
values are engineering targets chosen from the published reference points;
adjust them only by editing this table in a reviewed commit.

### 10.4 Required ablations at 50M (each is one config flag)

Each ablation trains with the same data and step budget and reports suite mean
accuracy, ECE and latency. A component stays in the default config only if
removing it costs accuracy or calibration beyond seed noise (3 seeds).

1. loops: `R` fixed at 1 vs sampled (does deliberation help at all?)
2. no deep supervision; no truncation (full backprop); no input re-injection;
   no damped update; no sandwich norm
3. no Option Set Layer (options scored independently)
4. no null node (threshold abstention only)
5. no numeric channel
6. no scratch slots
7. workspace vs "flat" baseline: same parameter count spent on a deeper state
   encoder with options in the sequence (the Laya layout). This is the test of
   the central architectural claim. If the flat baseline wins at matched
   compute, say so and reconsider.
8. v1 components, one at a time, only if someone wants them back: MoE FFN,
   differential attention, memory modules.

---

## 11. Implementation tasks

Phases run in order. Inside a phase, tasks run in order unless marked parallel.

### Phase 0: baseline and safety fixes in v1 (small, do first)

**T-001 [Haiku]** `rhen/model.py:3199`. When `self._core_block.rhen is None`,
raise `RuntimeError("forward_rhen requires enable_rhen=True")` instead of
building a random pointer head. Test: construct `EOSModel` with
`enable_rhen=False`, call `forward_rhen`, expect the error.

**T-002 [Sonnet]** Add `tests/test_v1_known_structural_issues.py` with three
tests marked `xfail(strict=True)` that document A2, A3 and A5 on the tiny preset:
(a) assert that the hidden state at state position 0 after the recurrent loops
of `forward_rhen` is unchanged when the option text of a branch changes
(expected to fail today because of A3); (b) assert that the result for question
X is equal (1e-5) whether X is packed first or second among two questions
(expected to fail today because of A2 and A3); (c) assert equal probabilities
per option when the option order is reversed (expected to fail today, A5).
A1 needs a state longer than the window, so add a fourth test with
`swa_window_size=8` and a 32-token state: changing the last state token must
change the layer-1 hidden vector of the first state token (expected to fail
today). These findings come from reading the code, not from running it, so if
any test unexpectedly passes, stop and report it instead of forcing it to fail.
The tests are regression documentation; the defects are not to be fixed in v1.

**T-003 [Haiku]** README: move the three arXiv badges to a "References"
section; add a line under Known limitations pointing to this plan and to the
three xfail tests.

### Phase 1: RHEN-2 skeleton

**T-101 [Haiku]** Create `rhen/v2/__init__.py`, `rhen/v2/config.py` with
`Rhen2Config` and `PRESETS` exactly as Section 5 plus `r2-tiny`. Add
`count_parameters(cfg)` that computes the table's formulas. Test: computed
totals for the four presets are within 2% of 52.4M, 318.3M, 1012M, 7968M.

**T-102 [Sonnet]** `rhen/v2/layers.py`: `RMSNorm` (import from `rhen.model`),
`SwiGLU`, `SelfAttention(d, heads, window=None)`, `CrossAttention(d, heads)`,
`FactorisedEmbedding`, `NumericChannel`, rotary helper with per-group
positions. Bidirectional; padding masks; `F.scaled_dot_product_attention`.
Tests: shapes; a padded batch gives the same output for real tokens as the
unpadded input (1e-5); window mask blocks tokens outside the window.

**T-103 [Sonnet]** `rhen/v2/packing.py`: `QuestionSpec`, `Rhen2Batch`,
`Rhen2Packer.pack(examples) -> Rhen2Batch`, number and date detection,
bucketing collator, sanitisation reuse. Tests: round-trip decode; spoofed
delimiters in the state are neutralised; numeric features are non-zero only on
number tokens; K and Lo limits enforced with clear errors.

### Phase 2: the model

**T-201 [Sonnet]** `rhen/v2/state_encoder.py` per 4.4. Tests: output depends
only on state inputs; changing a pad token does nothing; local/global schedule
honoured.

**T-202 [Sonnet] [Review]** `rhen/v2/workspace.py`: `CoreBlock`, `CoreStack`,
`OptionSetLayer`, `DeliberationWorkspace` per 4.5 and 4.6, including sampled
loop counts, truncated backprop, damped update, re-injection, sandwich norm,
per-question halting at inference, evidence subset.

**T-203 [Sonnet]** `rhen/v2/readouts.py` per 4.7, all six types and the null node.
Tests: probabilities sum to 1 including null; score distribution is a valid
distribution with ordered thresholds; quantiles are monotone and inside
`[lo, hi]`; extract respects `i <= j <= i + max_span`.

**T-204 [Sonnet]** `rhen/v2/model.py`: `Rhen2Model`, `Rhen2Output`,
`LoopPlan`, `Interventions` per 4.8. No module-level mutable state.

**T-205 [Sonnet]** `rhen/v2/losses.py` per 7.4. Tests: each task loss is
minimised by the true distribution on a toy 3-class problem (numerical check);
soft-label path works.

**T-206 [Haiku]** `tests/v2/test_invariance.py`: for `r2-tiny` with random
weights, permuting options permutes the probabilities exactly (max abs
difference under 1e-5) for choice, multi and noul; the null column does not move.

**T-207 [Haiku]** `tests/v2/test_isolation.py`: (a) the output for question X
is identical whether it is evaluated alone or with other questions in any
order; (b) changing any question or option leaves `S` bit-identical;
(c) two concurrent threads calling the model give the same results as
sequential calls; (d) gradient of question X's loss with respect to another
question's tokens is exactly zero.

**T-208 [Haiku]** `tests/v2/test_overfit.py`: `r2-tiny` reaches over 95%
accuracy on 64 synthetic examples from `rhen/research/tasks.py` within 300
steps on CPU. This is the "the model can learn at all" smoke test v1 never had.

### Phase 3: data

**T-301 [Sonnet]** `rhen/v2/data/schema.py` (example record, JSONL reader and
writer compatible with `DecisionExample`) and `converters/` for the first six
datasets: BoolQ, MNLI, AG News, Banking77, SQuAD 2.0, GoEmotions. Each
converter has a 20-example fixture test and records the licence string.

**T-302 [Haiku]** Remaining converters from 7.1.2 (parallel, one per dataset).

**T-303 [Sonnet]** `rhen/v2/data/synth.py`: 40 programmatic families with
in-distribution and OOD variants, deterministic by seed.

**T-304 [Haiku]** `blocklist.json` and the loader check; test that a
blocklisted example raises.

**T-305 [Sonnet] [Review]** `rhen/v2/data/teacher.py`: generation and
soft-labelling scripts with a provider-agnostic interface, resumable JSONL
output, cost logging, and a `--dry-run` mode with a fake teacher for tests.
No API keys in the repository.

**T-306 [Sonnet]** `rhen/v2/data/pretrain.py`: MLM masking, span-choice and
span-locate example builders per 7.2.

### Phase 4: training

**T-401 [Sonnet]** `rhen/v2/train.py`: reuse `rhen/research/` plumbing
(device, distributed, checkpointing, manifest, metrics). Batched forward, mixed
objectives by ratio, EMA weights, loop sampling, LR schedule, muP scaling
option, resumable. Config files `configs/v2/r2_tiny_cpu.json`,
`r2_50m_pretrain.json`, `r2_50m_decision.json`.

**T-402 [Haiku]** CLI `rhen2-train`, `rhen2-eval` entries in `pyproject.toml`.

**T-403 [Sonnet]** End-to-end CPU test: pretrain `r2-tiny` 200 steps, decision
train 300 steps, evaluate, save, resume, results identical after resume.

### Phase 5: calibration and Checker

**T-501 [Sonnet]** `rhen/v2/calibrate.py` per Section 8, writing and loading
`calibration.json` with hash checks. Tests on synthetic logits with known
miscalibration: fitted temperature recovers the true one within 5%; group
conformal coverage within 2 points of target on 20k synthetic draws.

**T-502 [Sonnet] [Review]** `rhen/v2/checker.py` per Section 6: C1 head and
features, C2 evidence, C3 interventions through `Interventions`, C4 using
`CheapVerifier`, verdict logic, `CheckedDecision`.

**T-503 [Sonnet]** Cross-fitting script for C1 and the selective fine-tuning
step (7.6). Test on a toy model where half the questions are unanswerable:
`p_correct` AUROC above 0.9.

**T-504 [Haiku]** Tests for each flag: construct a tiny model and a state where
the answer token is the only evidence; verify C3a lowers the probability when
it is masked; verify C3b flags a question whose answer is given in the
question text itself.

### Phase 6: pipeline consolidation

**T-601 [Sonnet] [Review]** `rhen/v2/pipeline.py`: `DecisionPipeline`,
`DecisionTrace`, stages 1 to 10, fail-closed rules, state cache (LRU keyed by
hash, size by bytes).

**T-602 [Sonnet]** Policy per 9.4 in `rhen/v2/policy.py`; keep
`PolicyStatus` values; migration shim so `DecisionPolicy(utility=...)` still
works with a deprecation warning.

**T-603 [Sonnet]** Wire `RhenEngine`, `UnifiedAIOrchestrator` and
`ForecastAwareDecider` to accept `backend="v2"` and delegate. Default stays
`v1` until the 50M gates pass, then flips in one commit.

**T-604 [Haiku]** Feedback and drift per 9.6; bounded `decision_history`.

**T-605 [Haiku]** Hierarchical routing on model option vectors per 9.3.

### Phase 7: evaluation

**T-701 [Sonnet]** `rhen/v2/eval.py` and `benchmarks/suite/` per Section 10,
with the numpy baselines from the Banking77 harness run on every dataset.

**T-702 [Haiku]** `benchmarks/latency/` script: cold and warm, by `Ls` and `K`.

**T-703 [Sonnet]** Ablation runner: one command per ablation in 10.4, results
table generated from the JSON files.

### Phase 8: scale-up (after training results exist)

**T-801** 300M: apply 5.1 "50M -> 300M" items 1 to 6. **T-802** 1B: apply
"300M -> 1B" items 1 to 5. **T-803** 8B: decide (a) or (b) with the user, then
apply items 1 to 4. Each is a config and training-setup change plus a gate run;
no architecture code changes are expected. If one is needed, stop and revise
this plan first.

### Phase 9: cleanup (after v2 is the default)

Remove the root shim modules and the duplicate `optim/` and `eos_research/`
packages in a major version; reduce CI to one full run plus a smoke matrix;
move v1 model code to `rhen/legacy/`.

---

## 12. Risks and open questions

1. **Data is the bottleneck.** Teacher labelling at the sizes in 7.1.4 costs
   real money and inherits teacher errors. Mitigation: teacher ensembles, keep
   disagreement as soft labels, and audit 1,000 labels by hand per domain.
2. **Looped training instability.** Known failure modes: state collapse, the
   model ignoring the recurrent state, overshoot. Mitigation is the recipe in
   4.5; monitor accuracy at `R=1` vs `R=r_max` every evaluation. If they are
   equal, the loops are not being used.
3. **The workspace may not beat a flat encoder** at equal compute. Ablation 7
   in 10.4 decides this early and cheaply. The pipeline, Checker, calibration
   and data work are all valid with either trunk.
4. **Attention-based evidence may be unfaithful.** C3a measures it. If the
   deletion effect is weak, switch C2 to gradient-times-input attribution.
5. **8B from scratch is likely out of budget.** Option (b) in 5.1 is the
   practical route and matches what Microsoft shipped.
6. **Decisions for the user:** English-only or multilingual (affects the
   tokenizer before 1B); which teacher models and budget; whether API access
   to JEV or Microsoft-Decision-1 is available for head-to-head runs; target
   hardware for latency gates.
