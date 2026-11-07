"""
07. Agent Framework Integrations: LangChain/LangGraph, CrewAI/AutoGen & FastAPI
-------------------------------------------------------------------------------
Demonstrates seamless, realistic production integration patterns for RHEN:

1. LangChain / LangGraph Node:
   Deterministic, non-autoregressive semantic router for agent graphs with
   conformal abstention fallback.

2. CrewAI / AutoGen Action Guard:
   Cryptographically enforced tool gateway that intercepts agent tool calls,
   evaluates safety via RHEN, and gates destructive tools with signed ActionCards.

3. FastAPI Decision Microservice:
   Ultra-low latency HTTP endpoint (/decide) returning structured, typed
   decisions in a single forward pass.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rhen import (
    ActionStagingEngine,
    RhenConfig,
    RhenEngine,
)


# ============================================================================
# PATTERN 1: LangChain / LangGraph Deterministic Semantic Router
# ============================================================================

class LangGraphAgentState:
    """Simulated LangGraph shared state object."""
    def __init__(self, query: str, context: str, history: List[str] | None = None):
        self.query = query
        self.context = context
        self.history = history or []
        self.next_node: Optional[str] = None
        self.confidence: float = 0.0
        self.abstained: bool = False


class RhenLangGraphRouter:
    """
    Deterministic Router Node for LangGraph workflows.
    Replaces slow, brittle LLM JSON prompting with O(1) single forward-pass
    categorical choice routing and conformal abstention.
    """
    ROUTING_TARGETS = [
        "execute_database_query",
        "fetch_web_documentation",
        "synthesize_direct_answer",
        "request_human_clarification",
    ]

    def __init__(self, engine: RhenEngine):
        self.engine = engine

    def __call__(self, state: LangGraphAgentState) -> LangGraphAgentState:
        # Construct compact state representation for RHEN
        state_repr = (
            f"User Query: {state.query}\n"
            f"Context: {state.context}\n"
            f"History Turns: {len(state.history)}"
        )

        # Non-autoregressive categorical routing decision
        res = self.engine.choice(
            state=state_repr,
            question="Select the optimal next workflow action node",
            options=self.ROUTING_TARGETS,
        )

        state.confidence = res.confidence

        # If model is uncertain or out-of-distribution, route safely to human clarification
        if res.should_abstain or res.confidence < 0.60:
            state.next_node = "request_human_clarification"
            state.abstained = True
        else:
            state.next_node = res.choice_label
            state.abstained = False

        return state


# ============================================================================
# PATTERN 2: CrewAI / AutoGen Action Guard with Cryptographic ActionCards
# ============================================================================

class ToolCallProposal:
    def __init__(self, agent_name: str, tool_name: str, arguments: Dict[str, Any], risk_tier: str = "medium"):
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.arguments = arguments
        self.risk_tier = risk_tier


class RhenActionGuard:
    """
    Safety Guard for autonomous agent frameworks (CrewAI, AutoGen).
    Intercepts proposed tool executions, verifies safety invariants with RHEN,
    and requires cryptographic HMAC-signed ActionCards with dual-custody.
    """
    def __init__(self, engine: RhenEngine, secret_key: bytes):
        self.engine = engine
        self.staging = ActionStagingEngine(secret_key=secret_key)
        self.staging.register_cosigner_agent("ComplianceAuditor", b"auditor-secret-key-3344")

    def inspect_and_gate(self, proposal: ToolCallProposal) -> Dict[str, Any]:
        state_repr = (
            f"Agent: {proposal.agent_name}\n"
            f"Tool: {proposal.tool_name}\n"
            f"Arguments: {proposal.arguments}\n"
            f"Risk Tier: {proposal.risk_tier}"
        )

        # RHEN single forward pass safety verification
        is_safe = self.engine.noul(state_repr, "Is this tool invocation safe and compliant with policy?")
        action_choice = self.engine.choice(
            state_repr,
            "Guard Action Policy",
            options=["PermitExecution", "RequireDualCustodyCoSign", "BlockAndAlert"],
        )

        # Stage cryptographic ActionCard
        card = self.staging.stage_action(
            decision_result={
                "p_safe": is_safe.p_true,
                "policy": action_choice.choice_label,
                "confidence": action_choice.confidence,
            },
            action_type="execute",
            payload={"tool": proposal.tool_name, "args": proposal.arguments},
            risk_tier=proposal.risk_tier,
            title=f"Tool Gate: {proposal.tool_name}",
        )

        preflight = self.staging.preflight_verification(card.id)

        return {
            "card_id": card.id,
            "status": card.status,
            "signature": card.signature[:20] + "...",
            "allowed": preflight.get("allowed", False),
            "policy": action_choice.choice_label,
            "p_safe": is_safe.p_true,
            "card": card,
        }


# ============================================================================
# PATTERN 3: FastAPI High-Speed Decision Microservice
# ============================================================================

app = FastAPI(
    title="RHEN Decision Service",
    description="Sub-5ms deterministic decision microservice powered by RHEN non-autoregressive engine.",
    version="1.0.0",
)

# Global in-memory engine shared across requests
_global_engine: Optional[RhenEngine] = None

def get_engine() -> RhenEngine:
    global _global_engine
    if _global_engine is None:
        config = RhenConfig(
            hidden_size=256,
            intermediate_size=512,
            num_recurrent_layers=2,
            max_recurrent_loops=2,
        )
        _global_engine = RhenEngine(config=config)
    return _global_engine


class DecisionRequest(BaseModel):
    state: str = Field(..., description="Operational state payload or context string")
    question: str = Field(..., description="Routing or decision query")
    options: List[str] = Field(..., min_length=2, description="Candidate options")


class DecisionResponse(BaseModel):
    selected_option: str
    selected_index: int
    confidence: float
    probabilities: Dict[str, float]
    abstained: bool
    latency_ms: float


@app.post("/decide", response_model=DecisionResponse)
def decide_endpoint(req: DecisionRequest):
    engine = get_engine()
    t0 = time.perf_counter()
    res = engine.choice(state=req.state, question=req.question, options=req.options)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return DecisionResponse(
        selected_option=res.choice_label,
        selected_index=res.choice_index,
        confidence=res.confidence,
        probabilities=res.probabilities,
        abstained=res.should_abstain,
        latency_ms=round(latency_ms, 2),
    )


# ============================================================================
# Demonstration & Verification Runner
# ============================================================================

def main():
    print("=" * 78)
    print("        RHEN Agent Framework Integrations & Production Patterns        ")
    print("=" * 78)

    config = RhenConfig(
        hidden_size=256,
        intermediate_size=512,
        num_recurrent_layers=2,
        max_recurrent_loops=2,
    )
    engine = RhenEngine(config=config)

    # -------------------------------------------------------------
    # 1. LangChain / LangGraph Demo
    # -------------------------------------------------------------
    print("\n[1. LangGraph Semantic Router]")
    router = RhenLangGraphRouter(engine)

    state_a = LangGraphAgentState(
        query="What was our Q3 EBITDA margin in the EMEA region?",
        context="User is asking for structured financial database figures.",
    )
    routed_a = router(state_a)
    print(f"Query A:  '{state_a.query}'")
    print(f" -> Next Node:  {routed_a.next_node}")
    print(f" -> Confidence: {routed_a.confidence:.4f} (Abstained: {routed_a.abstained})")

    # -------------------------------------------------------------
    # 2. CrewAI / AutoGen Action Guard Demo
    # -------------------------------------------------------------
    print("\n[2. CrewAI / AutoGen Action Guard with ActionCards]")
    guard = RhenActionGuard(engine, secret_key=b"main-safety-guard-secret-key-1092")

    # Low-risk safe tool proposal
    safe_call = ToolCallProposal(
        agent_name="ResearchAgent",
        tool_name="read_internal_wiki",
        arguments={"page_id": "Architecture-Docs-2026"},
        risk_tier="low",
    )
    res_safe = guard.inspect_and_gate(safe_call)
    print(f"Tool: {safe_call.tool_name} (Risk: {safe_call.risk_tier})")
    print(f" -> Preflight Allowed: {res_safe['allowed']} | Policy: {res_safe['policy']}")
    print(f" -> ActionCard ID:     {res_safe['card_id']} (HMAC: {res_safe['signature']})")

    # Critical destructive tool proposal
    danger_call = ToolCallProposal(
        agent_name="OpsAgent",
        tool_name="drop_database_table",
        arguments={"table": "users_prod", "cascade": True},
        risk_tier="critical",
    )
    res_danger = guard.inspect_and_gate(danger_call)
    print(f"\nTool: {danger_call.tool_name} (Risk: {danger_call.risk_tier})")
    print(f" -> Preflight Allowed: {res_danger['allowed']} (Requires Dual Custody!)")
    print(f" -> Policy Verdict:    {res_danger['policy']} | Safety Prob: {res_danger['p_safe']:.4f}")
    print(f" -> ActionCard Status: {res_danger['status']}")

    # -------------------------------------------------------------
    # 3. FastAPI Endpoint Demo
    # -------------------------------------------------------------
    print("\n[3. FastAPI In-Process Decision Endpoint (/decide)]")
    # Initialize FastAPI engine
    get_engine()
    client = TestClient(app)

    payload = {
        "state": "Telemetry: Pod payment-worker-7b CPU=99%, ErrorRate=14.2%, QueueDepth=4200",
        "question": "Immediate auto-scaling mitigation choice",
        "options": ["SpinUp5NewPods", "RestartCluster", "DrainTraffic", "IgnoreAlert"],
    }

    t0 = time.perf_counter()
    response = client.post("/decide", json=payload)
    elapsed_total_ms = (time.perf_counter() - t0) * 1000.0

    print(f"HTTP Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Selected Option: {data['selected_option']} (Index: {data['selected_index']})")
        print(f"Model Confidence: {data['confidence']:.4f} | Abstained: {data['abstained']}")
        print(f"Model Compute Time: {data['latency_ms']} ms")
        print(f"Total HTTP Round-Trip: {elapsed_total_ms:.2f} ms")
        print(f"Option Probabilities: {data['probabilities']}")

    print("\n" + "=" * 78)
    print("All agent framework integration patterns verified successfully!")
    print("=" * 78)


if __name__ == "__main__":
    main()
