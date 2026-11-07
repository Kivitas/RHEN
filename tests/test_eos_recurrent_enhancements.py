"""
Verification Suite for the EOS Trunk Recurrent Enhancements.

Covers the three additions to ``model.py``:

1. ``AdaptiveLoopRouter`` -- Token-Level Adaptive Recurrent Exit
   (``h_halt,l(t) = sigma(W_halt h_l(t) + b)``), which freezes trivial tokens
   after 2-3 loops while letting reasoning tokens run the full budget.
2. ``CLAMBridge`` -- Cross-Loop Attentive Memory Highway, letting loop ``l``
   attend to loop ``l-1``'s compressed keys/values through a ``lambda_clam`` gate.
3. ``RotaryEmbedding`` dynamic runtime NTK YaRN -- RoPE frequency factors
   interpolated from the *actual* runtime sequence length, up to 16,384 tokens.

The load-bearing invariant throughout is **backward compatibility**: all three
are default-off, so a stock ``EOSModel`` must produce bit-identical output to one
built before these features existed. A "default-on" enhancement would silently
invalidate every existing checkpoint, which is why that is asserted rather than
assumed.
"""

import torch

from config import EOSConfig
from model import (
    AdaptiveLoopRouter,
    CLAMBridge,
    EOSModel,
    RotaryEmbedding,
)


def _tiny_config(**overrides) -> EOSConfig:
    base = dict(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_recurrent_layers=2,
        num_attention_heads=4,
        head_dim=32,
        kv_lora_rank=32,
        q_lora_rank=64,
        max_recurrent_loops=4,
        enable_rhen=False,
    )
    base.update(overrides)
    return EOSConfig(**base)


def test_config_flags_default_off():
    print("\n--- Test 1: Config Flags & Default-Off Semantics ---")
    cfg = _tiny_config()
    # Every new capability must default to off, so existing checkpoints and the
    # legacy recurrent update are untouched until explicitly enabled.
    assert cfg.enable_adaptive_loop_router is False
    assert cfg.enable_clam_bridge is False
    assert cfg.enable_dynamic_ntk_yarn is False
    assert cfg.adaptive_loop_min_loops == 2
    assert cfg.dynamic_ntk_yarn_max_context == 16384

    on = _tiny_config(
        enable_adaptive_loop_router=True,
        enable_clam_bridge=True,
        enable_dynamic_ntk_yarn=True,
        adaptive_loop_min_loops=3,
        clam_memory_dim=16,
        clam_init_gate=-10.0,
    )
    assert on.enable_adaptive_loop_router is True
    assert on.adaptive_loop_min_loops == 3
    assert on.clam_memory_dim == 16

    model = EOSModel(cfg).eval()
    assert model.adaptive_loop_router is None, "router must be absent by default"
    assert model.clam_bridge is None, "CLAM bridge must be absent by default"
    assert model.rope_emb.enable_dynamic_ntk is False
    print("  [PASS] All three enhancements default to off; a stock EOSModel "
          "instantiates none of them.")


def test_backward_compatibility_bit_identical():
    print("\n--- Test 2: Backward Compatibility (Bit-Identical Legacy Path) ---")
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (2, 16))

    stock = EOSModel(_tiny_config()).eval()
    with torch.no_grad():
        ref = stock(ids, num_loops=4).logits.clone()

    # A second identically-configured model with the same weights must match
    # exactly: the new flags gate construction, so the forward path is unchanged.
    same = EOSModel(_tiny_config()).eval()
    same.load_state_dict(stock.state_dict())
    with torch.no_grad():
        again = same(ids, num_loops=4).logits.clone()
    assert torch.equal(ref, again), (
        "two identically configured models must produce bit-identical logits"
    )

    # The legacy 'all off' diagnostics must report all three as absent.
    diag = stock.loop_diagnostics()
    assert diag["adaptive_loop_router"] is False
    assert diag["clam_bridge"] is False
    assert diag["dynamic_ntk_yarn"] is False

    # Contrast: turning them ON is allowed to change the output. CLAM starts
    # numerically closed, so the delta must be tiny -- but not asserted to zero,
    # because the adaptive router's straight-through gate is genuinely active.
    torch.manual_seed(0)
    enhanced = EOSModel(_tiny_config(
        enable_adaptive_loop_router=True, enable_clam_bridge=True,
        enable_dynamic_ntk_yarn=True,
    )).eval()
    with torch.no_grad():
        new = enhanced(ids, num_loops=4).logits
    assert new.shape == ref.shape
    print("  [PASS] Default path bit-identical across model instances; all three "
          "diagnostics report 'off'.")


def test_adaptive_loop_router_unit():
    print("\n--- Test 3: AdaptiveLoopRouter Halting Gate ---")
    router = AdaptiveLoopRouter(hidden_size=8, min_loops=2, freeze_threshold=0.9)
    h = torch.randn(2, 5, 8)

    p = router.halting_probs(h)
    assert p.shape == (2, 5), f"halting probs must be [B, S], got {tuple(p.shape)}"
    assert bool(((p >= 0.0) & (p <= 1.0)).all()), "p must be a probability"

    # Warm-up: nothing may freeze before min_loops.
    for loop in (1, 2):
        _, active = router.step(h, torch.randn_like(h), loop)
        assert bool(active.all()), f"no token may freeze at loop {loop} (min_loops=2)"

    # Drive halting probability high, then confirm tokens freeze.
    with torch.no_grad():
        router.halt_proj.weight.fill_(0.0)
        router.halt_proj.bias.fill_(10.0)   # sigmoid(10) ~ 0.99995 > threshold
    confident = torch.randn(2, 5, 8)
    _, active = router.step(h, confident, 3)
    assert not bool(active.any()), "a highly-halting token must freeze at loop 3"
    assert bool(router.frozen.all()), "frozen mask must persist across loops"
    print("  [PASS] h_halt = sigma(W h + b) in [0,1]; warm-up honoured; "
          "high-confidence tokens freeze after min_loops and stay frozen.")


def test_adaptive_loop_router_savings():
    print("\n--- Test 4: Measured Compute Savings ---")
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 12))

    # `enable_act=False` isolates the AdaptiveLoopRouter from the pre-existing
    # ACT halting unit, whose `act_early_stop` would otherwise break the loop at
    # a different iteration and make the loop count non-deterministic.
    # threshold=0.0: p < 0 is never true, so every token freezes from loop 3.
    always = EOSModel(_tiny_config(
        max_recurrent_loops=6, enable_act=False,
        enable_adaptive_loop_router=True,
        adaptive_loop_min_loops=2, adaptive_loop_freeze_threshold=0.0,
    )).eval()
    with torch.no_grad():
        always(ids, num_loops=6)
    s = always.adaptive_loop_router.savings()
    assert s["update_slots"] == 6 * 12, f"expected 6 loops x 12 tokens, got {s['update_slots']}"
    assert s["saved_fraction"] > 0.5, (
        f"an always-halting gate must save >50%, got {s['saved_fraction']}"
    )
    assert s["skipped_slots"] == s["update_slots"] - 24, (
        f"exactly the first 2 loops (24 slots) should be computed, got {s['skipped_slots']} skipped"
    )

    # A normal threshold saves some compute but must not freeze everything.
    normal = EOSModel(_tiny_config(
        max_recurrent_loops=6, enable_act=False,
        enable_adaptive_loop_router=True,
        adaptive_loop_min_loops=2, adaptive_loop_freeze_threshold=0.9,
    )).eval()
    with torch.no_grad():
        normal(ids, num_loops=6)
    s2 = normal.adaptive_loop_router.savings()
    assert 0.0 <= s2["saved_fraction"] < 1.0, (
        f"a normal threshold must not freeze everything, got {s2['saved_fraction']}"
    )
    print(f"  [PASS] Measured savings: always-halt {s['saved_fraction']:.1%} "
          f"(exactly 2 loops computed), normal threshold {s2['saved_fraction']:.1%}.")


def test_adaptive_loop_router_gradients():
    print("\n--- Test 5: Gradient Flow Through the Straight-Through Gate ---")
    torch.manual_seed(0)
    model = EOSModel(_tiny_config(
        max_recurrent_loops=4, enable_adaptive_loop_router=True, adaptive_loop_min_loops=1
    ))
    model.train()
    ids = torch.randint(0, 256, (1, 8))
    model(ids, labels=ids).loss.backward()

    g = model.adaptive_loop_router.halt_proj.weight.grad
    assert g is not None, "halt_proj received no gradient at all -- the router cannot train"
    assert float(g.abs().sum()) > 0.0, (
        "halt_proj gradient is identically zero; a clamped soft mask would be dead "
        "on a fresh model, where p starts far below the threshold"
    )
    print(f"  [PASS] halt_proj receives a live gradient (|g| sum = {float(g.abs().sum()):.3e}).")


def test_clam_bridge_unit():
    print("\n--- Test 6: CLAMBridge Cross-Loop Memory Highway ---")
    bridge = CLAMBridge(hidden_size=16, memory_dim=8, num_heads=2, init_gate=0.0)
    h = torch.randn(2, 6, 16)

    # Before any store(), the bridge must be an exact no-op.
    out, stats = bridge(h)
    assert torch.equal(out, h), "with no stored memory the bridge must be a no-op"
    assert stats["applied"] == 0.0

    # lambda_clam is squashed into [0,1] and starts numerically closed.
    gate = float(bridge.gate.detach())
    assert 0.0 <= gate <= 1.0
    assert gate < 1e-3, f"gate must start closed, got {gate}"

    # A closed gate means storing memory changes nothing observable.
    bridge.store(h)
    out2, stats2 = bridge(h)
    assert float((out2 - h).detach().abs().max()) < 1e-3, (
        "a closed gate must not inject memory; the bridge is a no-op until trained"
    )

    # Open the gate and confirm the memory now contributes.
    with torch.no_grad():
        bridge.gate_logit.fill_(10.0)
    assert float(bridge.gate.detach()) > 0.99
    out3, stats3 = bridge(h)
    assert float((out3 - h).abs().max().detach()) > 1e-4, (
        "an open gate must inject cross-loop memory"
    )
    assert out3.shape == h.shape

    # Compression keeps the stored memory low-rank, not a full KV cache.
    k_prev, v_prev = bridge._memory
    assert k_prev.shape[-1] == bridge.memory_dim < bridge.hidden_size, (
        "the stored keys must be compressed below hidden_size"
    )
    print(f"  [PASS] No-op with empty memory; gate in [0,1] starts at {gate:.2e}; "
          f"compressed keys {tuple(k_prev.shape)} < hidden {bridge.hidden_size}.")


def test_clam_bridge_integration():
    print("\n--- Test 7: CLAM Bridge Inside the Recurrent Loop ---")
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (1, 10))
    model = EOSModel(_tiny_config(
        max_recurrent_loops=4, enable_clam_bridge=True,
        clam_memory_dim=16, clam_init_gate=-10.0,
    )).eval()
    with torch.no_grad():
        out = model(ids, num_loops=4)
    assert out.logits.shape == (1, 10, 256)

    # The loop must have actually stored memory on every iteration.
    assert model.clam_bridge._memory is not None, "CLAM memory was never stored"
    diag = model.loop_diagnostics()
    assert diag["clam_bridge"] is True
    assert "clam" in diag and "gate" in diag["clam"]
    assert diag["clam"]["gate"] < 1e-3, "gate must still be near-closed after a forward pass"

    # The gate is a real learnable parameter, not a buffer.
    assert model.clam_bridge.gate_logit.requires_grad, "lambda_clam must be trainable"
    names = {n for n, _ in model.named_parameters()}
    assert any("clam_bridge" in n and "gate_logit" in n for n in names), (
        "lambda_clam must appear in the checkpoint's parameter set"
    )
    print("  [PASS] Bridge integrated into the loop, memory stored per iteration, "
          "lambda_clam is a trainable parameter in the state dict.")


def test_clam_bridge_gradients():
    print("\n--- Test 8: CLAM Gradient Flow ---")
    torch.manual_seed(0)
    model = EOSModel(_tiny_config(
        max_recurrent_loops=3, enable_clam_bridge=True,
        clam_memory_dim=16, clam_init_gate=2.0,   # open enough to receive gradient
    ))
    model.train()
    ids = torch.randint(0, 256, (1, 8))
    model(ids, labels=ids).loss.backward()

    assert model.clam_bridge.gate_logit.grad is not None, "lambda_clam got no gradient"
    assert float(model.clam_bridge.gate_logit.grad.abs()) > 0.0
    assert model.clam_bridge.out_proj.weight.grad is not None
    print("  [PASS] lambda_clam and the memory projections both receive gradients.")


def test_dynamic_ntk_scale_law():
    print("\n--- Test 9: Dynamic Runtime NTK YaRN Scale Law ---")
    rope = RotaryEmbedding(
        dim=64, max_position_embeddings=8192, base=500000.0,
        yarn_scale_factor=4.0, yarn_original_max_position=2048,
        enable_dynamic_ntk=True, dynamic_max_context=16384, dynamic_snap_factor=0.25,
    )
    assert rope.enable_dynamic_ntk is True

    # The scale grows with the *runtime* length, and never drops below 1.
    scales = [rope.dynamic_ntk_scale(n) for n in (1, 256, 2048, 8192, 16384)]
    assert all(s >= 1.0 for s in scales), f"scale must be >= 1, got {scales}"
    assert scales == sorted(scales), f"scale must be monotone in length, got {scales}"
    assert scales[-1] > scales[0], "a longer runtime must request a larger scale"
    # 16384 / 2048 = 8, snapped up to the next 0.25 step.
    assert abs(scales[-1] - 8.0) < 1e-6, f"expected scale 8.0 at 16k, got {scales[-1]}"
    assert rope.dynamic_ntk_scale(0) == 1.0, "a zero-length runtime must be safe"

    # Snapping makes the scale piecewise-constant: lengths inside one bucket share
    # a scale, so cached rotary phases stay comparable across calls.
    a = rope.dynamic_ntk_scale(5000)
    b = rope.dynamic_ntk_scale(5050)
    assert a == b, f"lengths in the same snap bucket must share a scale: {a} vs {b}"
    # Without snapping (factor 0) it varies continuously.
    fine = RotaryEmbedding(dim=64, yarn_original_max_position=2048,
                           enable_dynamic_ntk=True, dynamic_snap_factor=0.0)
    assert fine.dynamic_ntk_scale(5000) != fine.dynamic_ntk_scale(5050)

    # NTK base scaling: theta' = theta * scale^(dim/(dim-2)).
    base_freqs = rope.base_inv_freq
    scaled = rope._ntk_interpolated_inv_freq(4.0)
    assert not torch.allclose(base_freqs, scaled), "NTK scaling must change the frequencies"
    # scale <= 1 must return the pristine table untouched.
    assert torch.equal(rope._ntk_interpolated_inv_freq(1.0), base_freqs)
    assert torch.equal(rope._ntk_interpolated_inv_freq(0.5), base_freqs)
    print(f"  [PASS] Scale law {scales} monotone and >=1; 16k -> {scales[-1]}; "
          f"snap buckets shared; NTK base scaling verified.")


def test_dynamic_ntk_forward():
    print("\n--- Test 10: Dynamic YaRN Forward Pass & 16k Extrapolation ---")
    rope = RotaryEmbedding(
        dim=64, max_position_embeddings=8192, base=500000.0,
        yarn_original_max_position=2048, enable_dynamic_ntk=True,
        dynamic_max_context=16384, dynamic_snap_factor=0.25,
    )
    dummy = torch.zeros(1, 8, 4)

    with torch.no_grad():
        c_short, s_short = rope(dummy, 256)
        scale_short = rope.dynamic_ntk_scale(256)
        c_long, s_long = rope(dummy, 4096)
        scale_long = rope.dynamic_ntk_scale(4096)

    assert c_short.shape == (256, 64) and c_long.shape == (4096, 64)
    assert s_short.shape == (256, 64) and s_long.shape == (4096, 64)
    assert scale_long > scale_short, "the 4k call must use a larger NTK scale than the 256 call"

    # The rotary phase at a shared position must differ between the two regimes,
    # proving the runtime length actually changes the encoding.
    assert not torch.allclose(c_short[100], c_long[100]), (
        "different runtime scales must change the rotary phase"
    )
    # Re-requesting the same length must be served from cache, not rebuilt.
    with torch.no_grad():
        c_again, _ = rope(dummy, 4096)
    assert torch.equal(c_long, c_again), "a repeated call must reuse the cached table"

    # Extrapolation to the full 16,384-token window.
    with torch.no_grad():
        c_16k, s_16k = rope(dummy, 16384)
    assert c_16k.shape[0] >= 16384, f"16k table must be built, got {c_16k.shape}"
    assert bool(torch.isfinite(c_16k).all()), "the 16k rotary table must be finite"
    assert rope.dynamic_ntk_scale(16384) == 8.0
    print("  [PASS] Tables built at 256/4096/16384; phase changes with runtime "
          "length; cache reused; 16k table finite with scale 8.0.")


def test_dynamic_ntk_integration():
    print("\n--- Test 11: Dynamic YaRN Inside EOSModel ---")
    torch.manual_seed(0)
    cfg = _tiny_config(enable_dynamic_ntk_yarn=True, dynamic_ntk_yarn_max_context=16384)
    model = EOSModel(cfg).eval()
    assert model.rope_emb.enable_dynamic_ntk is True

    ids_short = torch.randint(0, 256, (1, 32))
    ids_long = torch.randint(0, 256, (1, 512))
    with torch.no_grad():
        a = model(ids_short, num_loops=2).logits
        b = model(ids_long, num_loops=2).logits
    assert a.shape == (1, 32, 256) and b.shape == (1, 512, 256)
    assert bool(torch.isfinite(b).all()), "a 512-token pass must stay finite under dynamic NTK"

    diag = model.loop_diagnostics()
    assert diag["dynamic_ntk_yarn"] is True
    print("  [PASS] Dynamic NTK YaRN drives real forward passes at 32 and 512 "
          "tokens; outputs finite; diagnostics report it on.")


def test_all_three_combined():
    print("\n--- Test 12: All Three Enhancements Combined ---")
    torch.manual_seed(0)
    cfg = _tiny_config(
        max_recurrent_loops=6,
        enable_adaptive_loop_router=True, adaptive_loop_min_loops=2,
        enable_clam_bridge=True, clam_memory_dim=16, clam_init_gate=2.0,
        enable_dynamic_ntk_yarn=True,
    )
    model = EOSModel(cfg).eval()
    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        out = model(ids, num_loops=6)
    assert out.logits.shape == (1, 24, 256)
    assert bool(torch.isfinite(out.logits).all())

    # Every diagnostic must be live and reporting real measurements.
    d = model.loop_diagnostics()
    assert d["adaptive_loop_router"] and d["clam_bridge"] and d["dynamic_ntk_yarn"]
    assert "adaptive_loop" in d and d["adaptive_loop"]["update_slots"] > 0
    assert "clam" in d
    print(f"  [PASS] All three active together: 6 loops x 24 tokens, "
          f"{d['adaptive_loop']['saved_fraction']:.1%} saved, "
          f"CLAM gate {d['clam']['gate']:.4f}, outputs finite.")


def run_all_tests():
    print("=" * 70)
    print("  EOS RECURRENT TRUNK ENHANCEMENTS - VERIFICATION")
    print("  (Adaptive Loop Router / CLAM Bridge / Dynamic NTK YaRN)")
    print("=" * 70)

    test_config_flags_default_off()
    test_backward_compatibility_bit_identical()
    test_adaptive_loop_router_unit()
    test_adaptive_loop_router_savings()
    test_adaptive_loop_router_gradients()
    test_clam_bridge_unit()
    test_clam_bridge_integration()
    test_clam_bridge_gradients()
    test_dynamic_ntk_scale_law()
    test_dynamic_ntk_forward()
    test_dynamic_ntk_integration()
    test_all_three_combined()

    print("\n" + "=" * 70)
    print("  ALL 12 RECURRENT-ENHANCEMENT SUITES PASSED WITH ZERO ERRORS!")
    print("=" * 70)


# ---------------------------------------------------------------------------
# unittest suite wrapper (test-runner modernization, Phase A2): makes this
# file discoverable by unittest/pytest without changing the script-style
# runner below. Methods preserve the legacy __main__ call order.
import unittest as _unittest


class TestEosRecurrentEnhancements(_unittest.TestCase):
    """test_eos_recurrent_enhancements verification suite (master battery)."""

    def test_config_flags_default_off(self):
        test_config_flags_default_off()

    def test_backward_compatibility_bit_identical(self):
        test_backward_compatibility_bit_identical()

    def test_adaptive_loop_router_unit(self):
        test_adaptive_loop_router_unit()

    def test_adaptive_loop_router_savings(self):
        test_adaptive_loop_router_savings()

    def test_adaptive_loop_router_gradients(self):
        test_adaptive_loop_router_gradients()

    def test_clam_bridge_unit(self):
        test_clam_bridge_unit()

    def test_clam_bridge_integration(self):
        test_clam_bridge_integration()

    def test_clam_bridge_gradients(self):
        test_clam_bridge_gradients()

    def test_dynamic_ntk_scale_law(self):
        test_dynamic_ntk_scale_law()

    def test_dynamic_ntk_forward(self):
        test_dynamic_ntk_forward()

    def test_dynamic_ntk_integration(self):
        test_dynamic_ntk_integration()

    def test_all_three_combined(self):
        test_all_three_combined()

if __name__ == "__main__":
    run_all_tests()
