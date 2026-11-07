"""
Unified AI Orchestrator for EOS LLM.

Combines:
- Rhen decision layer: a single typed decision layer (AdvancedRhenEngine) for
  non-autoregressive latent decisions, with speculative cascades
- Meta-Cognitive Monitoring: ECE calibration, Platt scaling, uncertainty decomposition
- Agent Tools: Action Card preflight verification and execution staging (Laya / Open-Jev)
- Context Builder: Multimodal and tabular ingestion (Open-Jev tabular & Laya entities)

The former three-layer pipeline (deliberative reasoning and creative generation) has
been removed; generation requests are rejected with NotImplementedError.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch

try:
    from rhen.rhen import AdvancedRhenEngine
    from rhen.config import EOSConfig
    from rhen.meta_cognitive_monitor import MetaCognitiveMonitor
    from rhen.agent_tools import ActionCard, ActionStagingEngine
    from rhen.context_builder import ContextBuilder
    from rhen.rhen_pillars import PillarStack, RhenPillarConfig
except ImportError:
    from rhen import AdvancedRhenEngine
    from config import EOSConfig
    from meta_cognitive_monitor import MetaCognitiveMonitor
    from agent_tools import ActionCard, ActionStagingEngine
    from context_builder import ContextBuilder
    from rhen_pillars import PillarStack, RhenPillarConfig


class UnifiedAIOrchestrator:
    """Unified AI Orchestrator for EOS LLM.

    Seamlessly integrates:
    1. The single Rhen decision layer (typed primitives, speculative cascades)
    2. Meta-Cognitive Calibration & Epistemic/Aleatoric Uncertainty Decomposition
    3. Action Card Staging with Preflight Security Checks
    4. Rich Multimodal and Cross-Platform Entity Context Construction
    """

    def __init__(
        self,
        config: EOSConfig,
        pillars: Optional[RhenPillarConfig] = None,
        action_signing_key: Optional[Union[bytes, str]] = None,
    ):
        """
        Args:
            config: Model and runtime configuration.
            pillars: Optional Rhen-1.0 pillar configuration (all inert by default).
            action_signing_key: Key that signs Action Cards. When omitted, the staging
                engine generates an ephemeral key for this orchestrator.
        """
        self.config = config

        # The single Rhen decision layer.
        self.semantic_engine = AdvancedRhenEngine(config=config)

        # Meta-cognitive monitor (calibration, ECE, Platt scaling, uncertainty decomposition)
        self.meta_cognitive = MetaCognitiveMonitor(config)
        self.meta_monitor = self.meta_cognitive
        self.adaptive_conformal = getattr(self.meta_monitor, "adaptive_conformal", None)

        # Action Staging Engine for Agent Tools & Preflight Guardrails
        self.action_staging = ActionStagingEngine(
            secret_key=action_signing_key,
            ephemeral_key=action_signing_key is None,
        )
        self.staging_engine = self.action_staging
        self.merkle_ledger = getattr(self.staging_engine, "merkle_ledger", None)
        self.bft_engine = getattr(self.staging_engine, "bft_engine", None)

        # Rhen-1.0 pillar stack (audit integrity, calibration routing, OOD
        # gates, search-before-RL). All pillars default to inert, so constructing
        # an orchestrator behaves exactly as before unless one is enabled.
        self.pillars = PillarStack(pillars)

    # -------------------------------------------------------------------------
    # Context Construction
    # -------------------------------------------------------------------------

    def create_context_builder(self) -> ContextBuilder:
        """Create a fresh ContextBuilder for structured multimodal context ingestion."""
        return ContextBuilder()

    # -------------------------------------------------------------------------
    # Decision Orchestration
    # -------------------------------------------------------------------------

    async def unified_decide(
        self,
        state: Any,
        questions: Optional[Union[List, Any]] = None,
        generation_request: Optional[str] = None,
        use_three_layer: Optional[bool] = None,
        context: Optional[Any] = None,
        auto_stage_actions: bool = False,
        latent_tensors: Optional[Dict[str, torch.Tensor]] = None,
        speculative_branches: Optional[Dict[int, List[Any]]] = None,
        feedback_label: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Unified decision interface: a speculative cascade, or typed decision primitives.

        ``use_three_layer`` is accepted for call compatibility and ignored; the
        three-layer pipeline was removed. ``generation_request`` is not supported and
        raises ``NotImplementedError``.
        """
        start_time = time.time()

        if generation_request is not None:
            raise NotImplementedError("generation layer removed")

        # If state is a ContextBuilder, extract its serialized string
        if isinstance(state, ContextBuilder):
            state = state.build_state_string()

        # Normalize questions list
        if questions is not None and not isinstance(questions, list):
            questions = [questions]

        result: Dict[str, Any] = {
            "orchestrator_mode": "unified_system_one",
            "timestamp": time.time(),
            "routing": {},
            "staged_actions": [],
        }

        # Speculative Cascade Mode (if speculative_branches is provided)
        if speculative_branches is not None:
            if not questions:
                raise ValueError("speculative_branches needs questions[0] as the root question")
            cascade_res = self.semantic_engine.speculative_cascade(
                state=state,
                root_question=questions[0],
                speculative_branches=speculative_branches,
                latent_tensors=latent_tensors,
            )
            result["orchestrator_mode"] = "speculative_cascade"
            result["routing"] = "speculative_cascade"
            result["speculative_cascade"] = cascade_res
            result["results"] = cascade_res.get("all_results", [cascade_res.get("root")])
            result["final_answers"] = [{
                "question_index": 0,
                "question_type": "SpeculativeCascade",
                "final_answer": cascade_res.get("root"),
                "winning_option_index": cascade_res.get("winning_option_index"),
                "active_branch": cascade_res.get("active_branch"),
                "speculative_cascade": cascade_res,
            }]

        # Decision Mode (typed primitives through the single Rhen decision layer)
        elif questions is not None:
            result.update(await self._decision_mode(state, questions, latent_tensors=latent_tensors))

        else:
            raise ValueError("unified_decide needs questions or speculative_branches")

        # Optional Auto-Staging of Action Cards with automatic Merkle ledger recording.
        # The risk tier defaults to None: the staging engine infers it from action and confidence.
        if auto_stage_actions and "final_answers" in result:
            auto_risk = kwargs.get("auto_stage_risk_tier")
            for item in result["final_answers"]:
                card = self.stage_action(
                    decision_result=item.get("final_answer", {}),
                    title=f"AutoAction-{item.get('question_type', 'Decision')}",
                    persona="AutoRemediator",
                    risk_tier=auto_risk,
                )
                result["staged_actions"].append(card.to_dict())

        # Update adaptive_conformal if feedback/label is present
        target_label = feedback_label
        if target_label is None:
            for k in ("label", "y_true", "true_label", "ground_truth", "feedback"):
                if k in kwargs and kwargs[k] is not None:
                    target_label = kwargs[k]
                    break

        if target_label is not None and hasattr(self.meta_monitor, "adaptive_conformal") and self.meta_monitor.adaptive_conformal is not None:
            pred_set: List[Any] = []
            if "results" in result and result["results"]:
                first_r = result["results"][0]
                conf_dict = getattr(first_r, "conformal_set", None) or (first_r.get("conformal_set") if isinstance(first_r, dict) else None)
                if conf_dict and isinstance(conf_dict, dict) and "prediction_set" in conf_dict:
                    pred_set = conf_dict["prediction_set"]
                elif hasattr(first_r, "choice_label") and first_r.choice_label:
                    pred_set = [first_r.choice_label]
                elif hasattr(first_r, "decision"):
                    pred_set = ["Yes" if first_r.decision else "No"]
                elif isinstance(first_r, dict):
                    if first_r.get("chosen_option"):
                        pred_set = [first_r["chosen_option"]]
                    elif "decision" in first_r:
                        pred_set = ["Yes" if first_r["decision"] else "No"]
            elif "final_answers" in result and result["final_answers"]:
                fa = result["final_answers"][0].get("final_answer", {})
                conf_dict = getattr(fa, "conformal_set", None) or (fa.get("conformal_set") if isinstance(fa, dict) else None)
                if conf_dict and isinstance(conf_dict, dict) and "prediction_set" in conf_dict:
                    pred_set = conf_dict["prediction_set"]
                elif hasattr(fa, "choice_label") and fa.choice_label:
                    pred_set = [fa.choice_label]
                elif hasattr(fa, "decision"):
                    pred_set = ["Yes" if fa.decision else "No"]
                elif isinstance(fa, dict):
                    if fa.get("chosen_option"):
                        pred_set = [fa["chosen_option"]]
                    elif "decision" in fa:
                        pred_set = ["Yes" if fa["decision"] else "No"]

            if not pred_set and target_label is not None:
                pred_set = [target_label]

            update_res = self.meta_monitor.adaptive_conformal.update(
                y_true=target_label,
                prediction_set=pred_set,
            )
            result["adaptive_conformal_update"] = update_res

        result["latency_ms"] = (time.time() - start_time) * 1000.0
        return result

    def unified_decide_sync(
        self,
        state: Any,
        questions: Optional[Union[List, Any]] = None,
        generation_request: Optional[str] = None,
        use_three_layer: Optional[bool] = None,
        context: Optional[Any] = None,
        auto_stage_actions: bool = False,
        latent_tensors: Optional[Dict[str, torch.Tensor]] = None,
        speculative_branches: Optional[Dict[int, List[Any]]] = None,
        feedback_label: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for unified_decide."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    lambda: asyncio.run(
                        self.unified_decide(
                            state=state,
                            questions=questions,
                            generation_request=generation_request,
                            use_three_layer=use_three_layer,
                            context=context,
                            auto_stage_actions=auto_stage_actions,
                            latent_tensors=latent_tensors,
                            speculative_branches=speculative_branches,
                            feedback_label=feedback_label,
                            **kwargs,
                        )
                    )
                )
                return future.result()
        else:
            return asyncio.run(
                self.unified_decide(
                    state=state,
                    questions=questions,
                    generation_request=generation_request,
                    use_three_layer=use_three_layer,
                    context=context,
                    auto_stage_actions=auto_stage_actions,
                    latent_tensors=latent_tensors,
                    speculative_branches=speculative_branches,
                    feedback_label=feedback_label,
                    **kwargs,
                )
            )

    async def _decision_mode(
        self,
        state: Any,
        questions: List,
        latent_tensors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Evaluate typed primitives on the single Rhen decision layer, then monitor them."""
        results = self.semantic_engine.evaluate(state, questions, latent_tensors=latent_tensors)
        decision: Dict[str, Any] = {
            "mode": "decision",
            "results": results,
            "routing": "single_decision_layer",
            "final_answers": [
                {
                    "question_index": i,
                    "question_type": type(q).__name__,
                    "final_answer": res,
                }
                for i, (q, res) in enumerate(zip(questions, results))
            ],
        }
        decision["meta_cognitive"] = await self.meta_cognitive.monitor_decision(decision, state)
        return decision

    # -------------------------------------------------------------------------
    # Action Staging & Tool Verification Interfaces (Laya / Open-Jev style)
    # -------------------------------------------------------------------------

    def stage_action(
        self,
        decision_result: Any,
        action_type: str = "execute",
        payload: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        persona: str = "DefaultAgent",
        risk_tier: Optional[str] = None,
        required_permissions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionCard:
        """Stage a new Action Card with preflight verification criteria and Merkle ledger recording.

        ``metadata`` is forwarded verbatim so decision certificates (e.g. the conformal
        prediction set ``C(S)`` and free-energy OOD verdict) reach the preflight gates.
        """
        card = self.action_staging.stage_action(
            decision_result=decision_result,
            action_type=action_type,
            payload=payload,
            title=title,
            persona=persona,
            risk_tier=risk_tier,
            required_permissions=required_permissions,
            metadata=metadata,
        )
        if hasattr(self.staging_engine, "merkle_ledger") and self.staging_engine.merkle_ledger is not None:
            if "merkle_leaf_index" not in card.metadata:
                merkle_entry = self.staging_engine.merkle_ledger.append_action(card)
                card.metadata["merkle_leaf_index"] = merkle_entry["leaf_index"]
                card.metadata["merkle_leaf_hash"] = merkle_entry["leaf_hash"]
                card.metadata["merkle_root"] = merkle_entry["merkle_root"]
        return card

    def verify_action(
        self,
        card: ActionCard,
        granted_permissions: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """Verify preflight permissions and safety thresholds for an Action Card."""
        return self.action_staging.verify_action(card, granted_permissions=granted_permissions)

    # -------------------------------------------------------------------------
    # Rhen-1.0 pillar integration
    # -------------------------------------------------------------------------

    def audit_decision(
        self,
        question: str,
        chosen: Any,
        confidence: float,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        """Seal a decision in the out-of-band audit ledger.

        Returns ``None`` when auditing is disabled (the default). Raises
        ``AuditSinkError`` when the sink is unreachable and the ledger is
        fail-closed -- a decision that could not be audited must not be reported
        as audited (arXiv:2609.30266).
        """
        if not self.pillars.config.enable_audit:
            return None
        entry = self.pillars.authorize_action(question, chosen, confidence, **extra)
        return {
            "sequence": entry.sequence,
            "entry_hash": entry.entry_hash,
            "mac": entry.mac,
        }

    def verify_audit_chain(self) -> Dict[str, Any]:
        """Verify every sealed audit entry; empty result when disabled."""
        if self.pillars.audit is None:
            return {"enabled": False}
        return self.pillars.audit.verify_chain()

    def calibration_route(self, state: Any) -> Dict[str, Any]:
        """Language-aware calibration profile for a state string."""
        text = state.build_state_string() if isinstance(state, ContextBuilder) else str(state)
        return self.pillars.route_calibration(text)

    def check_ood_regression(
        self, predictions: Sequence[Any], labels: Sequence[Any], name: str = "candidate"
    ) -> Dict[str, Any]:
        """Gate a candidate change against the held-out OOD scoreboard."""
        return self.pillars.check_regression(predictions, labels, name).to_dict()

    def establish_ood_baseline(
        self, predictions: Sequence[Any], labels: Sequence[Any], name: str = "baseline"
    ) -> Dict[str, Any]:
        return self.pillars.establish_baseline(predictions, labels, name)

    def search_ceiling(self, sample_fn: Callable[[int], Tuple[Any, float]]) -> Dict[str, Any]:
        """Search-only baseline that any later RL result must beat."""
        return self.pillars.search_baseline(sample_fn)

    def pillar_cost_report(self) -> Dict[str, Any]:
        """Per-pillar latency/benefit accounting."""
        return self.pillars.cost_report()

    def approve_action(self, card_id: str, approver: str = "SystemSupervisor") -> Optional[ActionCard]:
        """Approve a staged Action Card for execution."""
        return self.action_staging.approve_action(card_id, approver=approver)

    def reject_action(self, card_id: str, reason: str = "Declined") -> Optional[ActionCard]:
        """Reject a staged Action Card."""
        return self.action_staging.reject_action(card_id, reason=reason)

    def execute_action(
        self,
        card_id: str,
        executor_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        bypass_verification: bool = False,
    ) -> ActionCard:
        """Execute an Action Card across the execution boundary.

        The engine records the execute event in the Merkle ledger exactly once and stores its
        leaf position on the card, so nothing is recorded here.
        """
        return self.action_staging.execute_action(
            card_id=card_id,
            executor_fn=executor_fn,
            bypass_verification=bypass_verification,
        )

    def get_staged_actions(self) -> List[ActionCard]:
        """Return all currently staged actions."""
        return self.action_staging.get_staged_cards()

    def get_action_history(self) -> List[ActionCard]:
        """Return history of all processed actions."""
        return self.action_staging.get_history()


RhenOrchestrator = UnifiedAIOrchestrator

__all__ = ["UnifiedAIOrchestrator", "RhenOrchestrator"]
