"""
01. Quickstart: RHEN Non-Autoregressive Decision Primitives
------------------------------------------------------------
Demonstrates evaluating all 7 typed decision primitives over structured state
in a single forward pass without autoregressive token generation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rhen import (
    RhenConfig,
    RhenEngine,
    Choice,
    Noul,
)

def main():
    print("=" * 60)
    print("RHEN Decision Engine - 7 Typed Primitives Quickstart")
    print("=" * 60)

    # Initialize a fast model configuration for local demonstration
    config = RhenConfig(
        hidden_size=256,
        intermediate_size=512,
        num_recurrent_layers=2,
        max_recurrent_loops=3,
    )
    engine = RhenEngine(config=config)

    state = (
        "Telemetry State:\n"
        "Node: srv-cluster-east-04\n"
        "CPU Load: 98.4%\n"
        "Replication Lag: 1420ms\n"
        "Disk I/O: saturated\n"
        "Active Alerts: HighLatency, DBReplicationLag\n"
        "Contact: ops-team@internal.corp"
    )
    print(f"\n[Input State]\n{state}\n")

    # 1. Noul: Binary Semantic Truth
    res_noul = engine.noul(state, "Is database replication severely degraded?")
    print(f"1. Noul: decision={res_noul.decision}, p_true={res_noul.p_true:.4f}, confidence={res_noul.confidence:.4f}")

    # 2. Choice: Categorical 1-of-K selection
    options = ["Throttle Read Queries", "Restart Replica Pod", "Failover to DR Region", "Do Nothing"]
    res_choice = engine.choice(state, "Recommended mitigation action", options)
    print(f"2. Choice: selected='{res_choice.choice_label}', index={res_choice.choice_index}, confidence={res_choice.confidence:.4f}")
    print(f"   Probabilities: {res_choice.probabilities}")

    # 3. Score: Ordered continuous spectrum (e.g. 1-10 severity)
    res_score = engine.score(state, "Incident severity level", min_score=1.0, max_score=10.0)
    print(f"3. Score: score={res_score.score:.2f}, levels={res_score.level_probabilities}")

    # 4. MultiChoice: Subset selection
    rules = ["PageOnCall", "QuarantineHost", "RateLimitIngress", "ArchiveLogs"]
    res_multi = engine.multichoice(state, "Applicable security/ops policies", rules, threshold=0.5)
    print(f"4. MultiChoice: selected={res_multi.selected_options}")

    # 5. Range: Bounded continuous interval regression (fitted Beta distribution)
    res_range = engine.range(state, "Estimated recovery time (minutes)", min_val=0.0, max_val=180.0)
    print(f"5. Range: expected={res_range.expected_value:.1f} mins, 95% interval=[{res_range.interval_low:.1f}, {res_range.interval_high:.1f}]")

    # 6. Extract: Span pointer extraction
    res_extract = engine.extract(state, slot_name="contact_email", query="Incident responder contact email")
    print(f"6. Extract: value='{res_extract.extracted_value}', span=({res_extract.start_idx}, {res_extract.end_idx})")

    # 7. Cascade: Multi-stage decision workflow
    stages = [
        Noul("Is critical cluster intervention required?"),
        Choice("Initial action", ["Automated Triage", "Manual Escalation"]),
    ]
    res_cascade = engine.cascade(state, stages=stages)
    print(f"7. Cascade: confidence={res_cascade.confidence:.4f}, decisions={res_cascade.decisions}")

    print("\n[SUCCESS] All 7 decision primitives evaluated successfully in single forward passes!")

if __name__ == "__main__":
    main()
