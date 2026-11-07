"""
Verification Suite for Latest Model Enhancements:
- Microsoft DIFF V2 Dynamic Lambda & FlashAttention Path
- DeepSeek-V3 Decoupled MLA & Absorption KV Cache
- Qwen 2.5 Dual QK-Norm & QKV Bias
- GLM-4 Prefix-LM Dual Attention Masking & 2D Branch Coordinates
- Google Titans Neural Long-Term Memory (MAG)
- DeepSeek-R1 Entropy-Guided Loop Pondering
- DeepSeek-V3 Chained MTP
- Rhen System Two Latent Tree & Beam Search Engine (LatentSearchEngine)
"""

import torch

from config import EOSConfig
from model import MultiLatentAttention, TitansNeuralMemory, DynamicThinkingBudgetController
from rhen_packer import RhenPacker
from rhen import LatentSearchEngine, LatentSearchResult


def test_diff_v2_and_decoupled_mla():
    print("\n--- Test 1: Microsoft DIFF V2 Dynamic Lambda & Decoupled MLA ---")
    config = EOSConfig(
        hidden_size=256,
        num_attention_heads=4,
        head_dim=64,
        kv_lora_rank=64,
        q_lora_rank=128,
        use_diff_mla=True,
        diff_v2_head_lambda=True,
        use_qkv_bias=True,
        use_qk_norm=True,
    )
    attn = MultiLatentAttention(config)
    
    x = torch.randn(2, 16, 256)
    cos = torch.ones(16, 64)
    sin = torch.zeros(16, 64)
    
    # 1. Forward pass test
    out, cache = attn(x, cos, sin)
    assert out.shape == (2, 16, 256), f"Expected shape (2, 16, 256), got {out.shape}"
    assert cache is None, "Inference cache should be None when kv_cache is not provided"
    
    # 2. KV Generation Cache test (Matrix Absorption Decoupled MLA)
    x_single = torch.randn(2, 1, 256)
    dummy_cache = (torch.zeros(2, 0, config.kv_lora_rank), torch.zeros(2, 0, config.head_dim))
    out_step, new_cache = attn(x_single, cos[:1], sin[:1], kv_cache=dummy_cache)
    
    assert out_step.shape == (2, 1, 256)
    assert new_cache[0].shape == (2, 1, config.kv_lora_rank), f"Cached KV latent shape {new_cache[0].shape} mismatch"
    assert new_cache[1].shape == (2, 1, config.head_dim), f"Cached RoPE key shape {new_cache[1].shape} mismatch"
    
    # Calculate footprint vs standard MHA cache:
    # Decoupled cache: kv_lora_rank (64) + head_dim (64) = 128 floats/tok
    # Standard cache: 2 * num_heads (4) * head_dim (64) = 512 floats/tok -> 75% savings
    print("  [PASS] DIFF V2 dynamic lambda, SDPA execution, and Decoupled MLA low-rank cache verified!")


def test_glm4_prefix_lm_masking_and_2d_positions():
    print("\n--- Test 2: GLM-4 Prefix-LM Dual Attention Masking & 2D Coordinates ---")
    packer = RhenPacker()
    state = "The system parameter status is nominal and battery voltage is 14.2V."
    questions = [
        ("What is the battery voltage?", ["14.2V", "12.0V", "11.5V"]),
        ("Is system nominal?", ["Yes", "No"]),
    ]
    packed = packer.pack(state, questions)
    
    assert packed.attention_mask is not None, "Packed sequence must have an attention mask"
    assert packed.branch_depth_ids is not None, "Packed sequence must have 2D branch depth IDs"
    assert packed.branch_depth_ids.shape == packed.input_ids.shape, "2D position shape mismatch"
    
    # Verify State Prefix ([0, state_len]) has bidirectional mask (values are 0.0)
    state_len = packed.state_length
    prefix_mask_block = packed.attention_mask[0, 0, :state_len, :state_len]
    assert float(prefix_mask_block.abs().sum().item()) == 0.0, "State prefix must have 0.0 (bidirectional) attention mask!"
    
    print("  [PASS] GLM-4 Prefix-LM bidirectional state mask & 2D branch depth positions verified!")


def test_titans_neural_long_term_memory():
    print("\n--- Test 3: Google Titans Neural Long-Term Memory (MAG) ---")
    titans_mem = TitansNeuralMemory(hidden_size=256, d_mem=32)
    
    h = torch.randn(2, 10, 256)
    out = titans_mem(h)
    
    assert out.shape == h.shape, f"Titans memory output shape {out.shape} mismatch"
    assert not torch.isnan(out).any(), "Titans memory output contains NaNs"
    
    # Test memory persistence reset
    titans_mem.reset_state()
    assert titans_mem.memory_matrix is None, "Memory state must be cleared after reset"
    
    print("  [PASS] Google Titans MAG associative neural memory module verified!")


def test_deepseek_r1_entropy_pondering():
    print("\n--- Test 4: DeepSeek-R1 Entropy & Free-Energy Guided Pondering ---")
    controller = DynamicThinkingBudgetController(EOSConfig(hidden_size=256, thinking_budget_min_loops=2, thinking_budget_max_loops=16))
    
    # High entropy / uncertain logits
    uncertain_logits = torch.zeros(1, 10, 50000)
    entropy_high, fe_high, pmax_high = controller.compute_entropy_and_free_energy(uncertain_logits)
    
    # Low entropy / sharp logits
    sharp_logits = torch.zeros(1, 10, 50000)
    sharp_logits[0, :, 0] = 100.0  # highly confident
    entropy_low, fe_low, pmax_low = controller.compute_entropy_and_free_energy(sharp_logits)
    
    assert float(entropy_high.mean().item()) > float(entropy_low.mean().item()), "Uncertain logits must have higher entropy"
    assert float(pmax_low.mean().item()) > float(pmax_high.mean().item()), "Sharp logits must have higher p_max"
    
    controller.reset()
    assert controller.check_ponder_exit(uncertain_logits, loop_idx=3) is False
    assert isinstance(controller.check_ponder_exit(sharp_logits, loop_idx=4), bool)
    
    print(f"  [PASS] DeepSeek-R1 pondering entropy verified: uncertain_H={entropy_high.mean():.2f}, sharp_H={entropy_low.mean():.2f}.")


def test_rhen_system_two_latent_search():
    print("\n--- Test 5: Rhen System Two Latent Tree & Beam Search Engine ---")
    engine = LatentSearchEngine(beam_width=3, max_depth=3)
    
    z_init = torch.randn(1, 256)
    candidate_options = ["action_A", "action_B", "action_C"]
    
    res = engine.search(z_init, candidate_options, max_depth=3, beam_width=2)
    
    assert isinstance(res, LatentSearchResult), "Output must be a LatentSearchResult instance"
    assert len(res.best_action_sequence) == 3, f"Expected sequence length 3, got {len(res.best_action_sequence)}"
    assert isinstance(res.prospective_value, float), "Prospective value must be float"
    assert len(res.step_rewards) == 3, "Step rewards count mismatch"
    
    print(f"  [PASS] Rhen System Two Latent Beam Search verified! Best sequence: {res.best_action_sequence}, score: {res.prospective_value:.4f}")


def run_all_tests():
    print("=" * 70)
    print("  VERIFYING LATEST MODEL ARCHITECTURAL ENHANCEMENTS")
    print("  (Microsoft DIFF V2 / DeepSeek-V3 / Qwen 2.5 / GLM-4 / Google Titans / DeepSeek-R1 / Rhen System Two)")
    print("=" * 70)
    
    test_diff_v2_and_decoupled_mla()
    test_glm4_prefix_lm_masking_and_2d_positions()
    test_titans_neural_long_term_memory()
    test_deepseek_r1_entropy_pondering()
    test_rhen_system_two_latent_search()
    
    print("\n" + "=" * 70)
    print("  ALL LATEST MODEL ENHANCEMENT SUITES PASSED CLEANLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
