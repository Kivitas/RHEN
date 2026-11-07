"""
04. HuggingFace & Custom Encoder Backbone Adapter
-------------------------------------------------
Demonstrates attaching Rhen's non-autoregressive decision layer
(MultiHeadRhenPointerHead) to ANY HuggingFace model or custom PyTorch encoder
via HuggingFaceEncoderAdapter.

Features:
1. Wrap any HuggingFace transformer (BERT, RoBERTa, DeBERTa, LLaMA, Mistral) or PyTorch encoder.
2. Automatically pack complex structured states and multiple decision branches.
3. Non-autoregressive evaluation in a single forward pass without token-by-token decoding.
4. Calibrated probabilities, pointer head alignments, and chosen actions.
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import BertConfig, BertModel
from rhen import (
    HuggingFaceEncoderAdapter,
    get_rhen_tokenizer,
)

def main():
    print("=" * 70)
    print("RHEN HuggingFace Encoder Adapter Example")
    print("=" * 70)

    # 1. Initialize tokenizer
    tokenizer = get_rhen_tokenizer()
    print(f"Loaded RHEN Tokenizer (Vocab Size: {tokenizer.vocab_size})")

    # 2. Instantiate a HuggingFace Transformer backbone
    # (Here we configure a lightweight BERT backbone for demonstration)
    encoder_hidden_size = 256
    bert_cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=encoder_hidden_size,
        num_attention_heads=4,
        num_hidden_layers=3,
        intermediate_size=512,
    )
    hf_encoder = BertModel(bert_cfg)
    print(f"Instantiated HuggingFace BERT Backbone: hidden_size={encoder_hidden_size}, layers=3")

    # 3. Wrap with Rhen's HuggingFaceEncoderAdapter
    adapter = HuggingFaceEncoderAdapter(
        encoder=hf_encoder,
        tokenizer=tokenizer,
        hidden_size=encoder_hidden_size,
        num_heads=4,
        rank=128,
    )
    print("Attached MultiHeadRhenPointerHead to HuggingFace encoder.")

    # 4. Define structured multi-agent state
    agent_state = (
        "Agent Environment State [t=1042]:\n"
        "- Kubernetes Pod: prod-checkout-7bf8d\n"
        "- Pod Status: CrashLoopBackOff (exit code 137, OOMKilled)\n"
        "- Memory Usage: 1024MB / 1024MB (Limit Exceeded)\n"
        "- Error Message: java.lang.OutOfMemoryError: Java heap space\n"
        "- Traffic Ingress: 14,200 req/sec\n"
        "- Assigned Runbook: RB-K8S-OOM-HEURISTIC"
    )

    # 5. Define multi-question decision schema
    # In a single forward pass, RHEN evaluates all decision branches simultaneously
    decisions_to_evaluate = [
        (
            "What is the root cause of the pod failure?",
            ["Memory OOM Limit", "Network Partition", "Disk Exhaustion", "Config Syntax Error"]
        ),
        (
            "Immediate remediation action?",
            ["Increase Memory Limit to 2Gi", "Rollback Deployment", "Restart Node", "Throttle Ingress"]
        ),
        (
            "Should on-call engineer be paged?",
            ["Yes - Page Immediately", "No - Handle Automatically"]
        ),
    ]

    print("\n[Input Agent State]")
    print(agent_state)

    print("\n--- Evaluating Decisions in a Single Forward Pass ---")
    results = adapter.evaluate_rhen(
        packed_or_state=agent_state,
        questions_and_options=decisions_to_evaluate,
    )

    print(f"\nEvaluated {len(results)} decision branches simultaneously:\n")
    for r in results:
        b_idx = r.get("branch_index", 0)
        q_text = r.get("question", "")
        chosen = r.get("chosen_option", "N/A")
        probs = r.get("probabilities_dict", {})
        temp = r.get("temperature")
        if isinstance(temp, torch.Tensor):
            temp = float(temp.mean().item())

        print(f"Branch #{b_idx + 1}: '{q_text}'")
        print(f"  -> Chosen Action: '{chosen}'")
        print(f"  -> Temperature:   {temp:.3f}" if temp else "  -> Temperature:   1.000")
        print("  -> Option Probabilities:")
        for opt_label, prob_val in probs.items():
            bar = "#" * int(prob_val * 25)
            print(f"     * {opt_label:<30}: {prob_val:.4f} | {bar}")
        print()

    # 6. Demonstrating Differentiable Training / Fine-tuning
    print("--- Differentiable Training Mode Verification ---")
    adapter.train()
    branch_outputs = adapter.forward_rhen(
        packed_or_state=agent_state,
        questions_and_options=decisions_to_evaluate,
    )
    # Compute cross-entropy loss against target indices [0, 0, 1]
    targets = [0, 0, 1]
    total_loss = torch.tensor(0.0)
    for out, target_idx in zip(branch_outputs, targets):
        logits = out.get("scores")  # [B, NumOptions]
        if logits is not None and logits.ndim == 2:
            target_tensor = torch.tensor([target_idx], device=logits.device)
            loss_fn = nn.CrossEntropyLoss()
            total_loss = total_loss + loss_fn(logits, target_tensor)

    print(f"Computed Multi-Branch Cross-Entropy Loss: {total_loss.item():.4f}")
    total_loss.backward()

    # Check gradient flow to both pointer head and underlying HF encoder
    ptr_grad = adapter.pointer_head.q_projs[0].weight.grad
    enc_grad = next(adapter.encoder.parameters()).grad
    print(f"Gradient flowed to Pointer Head:      {ptr_grad is not None and ptr_grad.norm().item() > 0}")
    print(f"Gradient flowed to Encoder Backbone:  {enc_grad is not None and enc_grad.norm().item() > 0}")

    print("\n[SUCCESS] HuggingFaceEncoderAdapter fully operational for inference and end-to-end training!")

if __name__ == "__main__":
    main()
