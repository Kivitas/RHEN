# Contributing to RHEN

Thank you for contributing to **RHEN** — the production-grade, non-autoregressive Joint Evaluation (JEV) decision engine for Agentic AI!

RHEN eliminates token generation overhead for agent tool orchestration and multi-agent control systems, replacing text-based JSON generation with typed semantic primitives, factorized bilinear pointer heads, conformal prediction sets, and cryptographic action cards.

---

## Developer Environment Setup

- **Step 1: Clone the Repository**:
   ```bash
   git clone https://github.com/your-org/rhen.git
   cd rhen
   ```

- **Step 2: Create and Activate a Virtual Environment**:
   ```bash
   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```

- **Step 3: Install Editable Package with Development Dependencies**:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -e ".[dev]"
   ```

---

## Code Architecture & Repository Layout

- **`rhen.py`**: High-level Pythonic SDK defining typed decision primitives (`Noul`, `Choice`, `Score`, `MultiChoice`, `Range`, `Extract`, `Cascade`), `RhenEngine`, and `AdvancedRhenEngine` (speculative cascades, latent search and calibrated-choice horizon projection).
- **`model.py`**: Recurrent looping transformer core with Differential Multi-Latent Attention (Diff-MLA), Shared-Expert MoE (SE-MoE), State-Space Memory Highway (SSM Bridge), and factorized bilinear pointer heads.
- **`rhen_packer.py`**: 2D block-causal sequence packaging, token escaping, and state delta caching (`<|state_diff|>`).
- **`agent_tools.py`**: HMAC-SHA256 signed `ActionCard`, tamper-evident Merkle action ledger, BFT quorum consensus, and cryptographic dual-custody co-signing.
- **`meta_cognitive_monitor.py`**: Split-conformal prediction sets $C(S)$, Helmholtz free-energy OOD detection, adaptive conformal martingale risk control, and temperature scaling.
- **`rhen_cognition.py`**: Cognitive extensions — RLCD policy optimization, Decision Duration Thresholding (TDT), hierarchical coarse-to-fine choice routing, and evidence lattices.
- **`rhen_pillars.py`**: Architectural pillar configuration and orchestrator bindings.
- **`rhen_verify.py` & `rhen_credit.py`**: Out-of-band audit trail socket receiver, geometry-based selectivity, and verifiable credit assignment.
- **`cli/`**: Entry-point implementations for `rhen-decide`, `rhen-eval`, `rhen-train`, and `rhen-tasks`.
- **`examples/`**: Self-contained, runnable production examples (primitives, action cards, conformal abstention, HuggingFace adapter, hierarchical routing, JEV benchmark, agent framework integrations).
- **`tests/`**: Comprehensive regression, mathematical verification, and framework test suites.

---

## Development & Verification Guidelines

1. **Code Quality & Linting** — all contributions must pass `pyflakes` with zero errors. CI checks all tracked Python files:
   ```bash
   python -m pyflakes $(git ls-files '*.py')
   ```
   In PowerShell:
   ```powershell
   python -m pyflakes (git ls-files '*.py')
   ```

2. **Run the Full Pytest Battery** — verify that all test suites pass with zero failures:
   ```bash
   pytest tests/
   ```
   You can also run isolated test suites:
   ```bash
   pytest tests/test_cli.py
   pytest tests/test_regressions_sdk.py
   pytest tests/test_regressions_infra.py
   ```

3. **Subprocess Test Battery Runner** — the root runner executes each test suite in an isolated subprocess:
   ```bash
   python run_tests.py              # Full battery
   python run_tests.py --fast       # Fast feedback loop (suites under 15s)
   python run_tests.py --slow       # Heavyweight verification gates
   python run_tests.py --suite test_cli.py
   ```

4. **Non-Autoregressive Decision Philosophy** — decisions must never require token-by-token generation. Decisions are evaluated in a single forward pass over packed 2D block-causal state and candidate options. Primitives must return typed result dataclasses (`NoulResult`, `ChoiceResult`, etc.) with explicit confidence and conformal sets.

5. **Backward Compatibility & Inert Defaults** — all new features and pillar controllers must be inert by default. Core models with default flags must retain exact numerical parity with existing checkpoints.

6. **Calibrated Confidence & Selective Abstention** — probabilities returned by RHEN must remain calibrated. Never rewrite confidence post-hoc. When state is off-manifold or uncertain, signal abstention through `should_abstain`, conformal sets $C(S)$, or the geometric convex-hull gate rather than guessing.

7. **Pull Request Checklist**
   - [ ] All tracked files pass pyflakes linting (`python -m pyflakes $(git ls-files '*.py')`).
   - [ ] Full test suite passes cleanly (`pytest tests/`).
   - [ ] New primitives or safety gates include unit tests in `tests/`.
   - [ ] No binary checkpoints (`.pt`, `.bin`), large datasets, or cache files committed.
   - [ ] Public API signatures and backward-compatible aliases remain intact.
