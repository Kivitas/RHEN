"""Deliberation workspace: a weight-shared recurrent block over question and option tokens.

The state is never re-read token by token. Question tokens (plus a few learned scratch slots)
and option tokens read the cached state memory by cross-attention, `R` times. Options see
only their own tokens, the state and their question; they interact with each other only in
the option-set layer, which has no positional information. That gives, by construction:

* a question's answer does not depend on any other question,
* the probabilities of the options are permuted exactly when the options are permuted,
* nothing is ever written into the state memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from rhen.v2.config import TYPE_INDEX, Rhen2Config
from rhen.v2.layers import Attention, AttnPool, EncoderBlock, RMSNorm, Rotary, SwiGLU
from rhen.v2.readouts import ReadoutOut

SCORE = TYPE_INDEX["score"]


@dataclass
class LoopPlan:
    n_loops: int = 4
    k_bptt: int = 2
    min_loops: int = 1
    halting: bool = False
    halt_threshold: float = 0.9
    train: bool = False

    @staticmethod
    def sample(cfg: Rhen2Config, generator: Optional[torch.Generator] = None) -> "LoopPlan":
        """Training plan: R = clamp(Poisson(r_mean - r_min) + r_min, r_min, r_max), drawn once per batch."""
        lam = float(cfg.r_mean - cfg.r_min)
        draw = int(torch.poisson(torch.tensor(lam), generator=generator).item()) if lam > 0 else 0
        r = max(cfg.r_min, min(cfg.r_max, draw + cfg.r_min))
        return LoopPlan(n_loops=r, k_bptt=cfg.k_bptt, min_loops=r, train=True)

    @staticmethod
    def effort(cfg: Rhen2Config, level: str, halt_threshold: float = 0.9) -> "LoopPlan":
        """Inference plan: low = r_min, medium = halting capped at 4, high = halting up to r_max."""
        if level == "low":
            return LoopPlan(n_loops=cfg.r_min, min_loops=cfg.r_min)
        if level == "medium":
            return LoopPlan(n_loops=min(4, cfg.r_max), min_loops=cfg.r_min, halting=True, halt_threshold=halt_threshold)
        if level == "high":
            return LoopPlan(n_loops=cfg.r_max, min_loops=cfg.r_min, halting=True, halt_threshold=halt_threshold)
        raise ValueError("effort must be 'low', 'medium' or 'high'")


@dataclass
class Interventions:
    """Counterfactual edits used by the Checker. They act on the memory the workspace reads."""
    memory_mask: Optional[torch.Tensor] = None    # [N, Ls] bool: replace these memory positions
    empty_memory: bool = False                    # keep only position 0 of the state
    null_delta: float = 0.0                       # extra score for the null option


class CoreBlock(nn.Module):
    def __init__(self, cfg: Rhen2Config):
        super().__init__()
        d = cfg.d_model
        self.n_self, self.n_cross, self.n_ffn = (RMSNorm(d, cfg.norm_eps) for _ in range(3))
        self.self_attn = Attention(d, cfg.n_heads, cfg.norm_eps)
        self.cross_attn = Attention(d, cfg.n_heads, cfg.norm_eps)
        self.ffn = SwiGLU(d, cfg.ffn_hidden)
        self.out_norm = RMSNorm(d, cfg.norm_eps) if cfg.use_sandwich_norm else None

    def forward(self, x, self_mask, rope, kv, kv_mask, per_group):
        x = x + self.self_attn(self.n_self(x), key_mask=self_mask, rope=rope)
        x = x + self.cross_attn(self.n_cross(x), kv=kv, key_mask=kv_mask, per_group=per_group)
        x = x + self.ffn(self.n_ffn(x))
        return self.out_norm(x) if self.out_norm is not None else x


class CoreStack(nn.Module):
    def __init__(self, cfg: Rhen2Config):
        super().__init__()
        self.layers = nn.ModuleList(CoreBlock(cfg) for _ in range(cfg.n_core_layers))

    def project(self, mem: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [layer.cross_attn.project_kv(mem) for layer in self.layers]

    def forward(self, x, self_mask, rope, kvs, kv_mask, per_group):
        for layer, kv in zip(self.layers, kvs):
            x = layer(x, self_mask, rope, kv, kv_mask, per_group)
        return x


class OptionSetLayer(nn.Module):
    """Mixes information across a question's options with no positional information.

    The set is [question, null, option_1 .. option_K]. Self-attention without positions is
    permutation-equivariant over the options, and the readout scores every option with shared
    weights, so permuting the options permutes the output probabilities.
    """

    def __init__(self, cfg: Rhen2Config):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.null_vec = nn.Parameter(torch.randn(d) * cfg.init_std)
        n_blocks = cfg.n_set_layers if cfg.use_set_layer else 0
        self.blocks = nn.ModuleList(EncoderBlock(d, cfg.n_heads, cfg.ffn_hidden, cfg.norm_eps) for _ in range(n_blocks))
        self.norm = RMSNorm(d, cfg.norm_eps)

    def forward(self, q, u, opt_valid):
        n, k, d = u.shape
        x = torch.cat([q[:, None], self.null_vec.expand(n, 1, d).to(u.dtype), u], dim=1)
        keep = torch.ones(n, 2, dtype=torch.bool, device=u.device)
        mask = torch.cat([keep, opt_valid], dim=1)
        if self.cfg.use_set_layer:
            for block in self.blocks:
                x = block(x, key_mask=mask)
            x = self.norm(x)
        return x[:, 0], x[:, 1], x[:, 2:]


class DeliberationWorkspace(nn.Module):
    def __init__(self, cfg: Rhen2Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.stack = CoreStack(cfg)
        self.scratch = nn.Parameter(torch.randn(cfg.n_scratch, d) * cfg.init_std)
        self.rank_emb = nn.Parameter(torch.zeros(cfg.max_levels, d))   # ordered levels of score questions
        self.step_logit = nn.Parameter(torch.zeros(()))
        self.q_pool, self.o_pool = AttnPool(d), AttnPool(d)
        self.set_layer = OptionSetLayer(cfg)
        self.rotary = Rotary(cfg.head_dim, cfg.rope_theta)

    @property
    def alpha(self) -> torch.Tensor:
        if not self.cfg.use_damping:
            return torch.ones((), device=self.step_logit.device)
        return torch.sigmoid(self.step_logit)

    def forward(
        self,
        S: torch.Tensor,                 # [B, Ls, d] shared memory, or [N, Ls, d] when per_question
        state_mask: torch.Tensor,        # matching mask
        q_emb: torch.Tensor,             # [N, Lq, d]
        q_mask: torch.Tensor,            # [N, Lq]
        o_emb: torch.Tensor,             # [N, K, Lo, d]
        o_mask: torch.Tensor,            # [N, K, Lo]
        opt_valid: torch.Tensor,         # [N, K]
        q_type: torch.Tensor,            # [N]
        *,
        per_question: bool,
        questions_per_example: int,
        plan: LoopPlan,
        readout: Callable[..., ReadoutOut],
        null_delta: float = 0.0,
    ):
        cfg = self.cfg
        n, _, d = q_emb.shape
        k, lo = o_emb.shape[1], o_emb.shape[2]
        m = questions_per_example
        ls = S.shape[1]
        dev = q_emb.device

        # Position 0 of every padded row stays real so no attention row is ever empty.
        q_mask = q_mask.clone()
        q_mask[:, 0] = True
        # Scratch slots come first so every real question token has a position that does not depend
        # on how far the batch happens to pad the questions.
        qmask_ext = torch.cat([torch.ones(n, cfg.n_scratch, dtype=torch.bool, device=dev), q_mask], dim=1)
        q0 = torch.cat([self.scratch.to(q_emb.dtype).expand(n, -1, -1), q_emb], dim=1)
        o_mask_flat = o_mask.clone()
        o_mask_flat[..., 0] = True
        o_mask_flat = o_mask_flat.reshape(n * k, lo)
        o0 = o_emb.reshape(n * k, lo, d)

        rope_q = self.rotary(q0.shape[1], dev, q0.dtype)
        rope_o = self.rotary(lo, dev, o0.dtype)
        rank_table = torch.zeros(k, d, device=dev, dtype=q0.dtype)
        used = min(k, cfg.max_levels)
        rank_table[:used] = self.rank_emb[:used].to(q0.dtype)
        rank = (q_type == SCORE).to(q0.dtype)[:, None, None] * rank_table[None]

        kvs_S = self.stack.project(S)
        per_group_q = 1 if per_question else m
        use_subset = cfg.evidence_n > 0 and k * lo * ls > cfg.evidence_budget and ls > cfg.evidence_n

        def per_q(t: torch.Tensor) -> torch.Tensor:
            """Give every question its own row of a memory-shaped tensor (a copy when shared)."""
            if per_question:
                return t
            return t[:, None].expand(t.shape[0], m, *t.shape[1:]).reshape(n, *t.shape[1:])

        state_mask_q = per_q(state_mask)
        b_of_q = torch.arange(n, device=dev) // (1 if per_question else m)

        def one_loop(Q, O, want_readout: bool, want_mass: bool = False):
            Qin = Q + q0 if cfg.use_reinjection else Q
            Oin = O + o0 if cfg.use_reinjection else O
            Qn = self.stack(Qin, qmask_ext, rope_q, kvs_S, state_mask, per_group_q)

            mass = None
            if use_subset or want_mass:
                last = self.stack.layers[-1].cross_attn
                if per_question:
                    mass = last.attention_mass(Qn, kvs_S[-1], state_mask, qmask_ext)
                else:
                    mass = self._mass_shared(last, Qn, kvs_S[-1], state_mask, qmask_ext, m)

            if use_subset:
                score = mass.masked_fill(~state_mask_q, -float("inf"))
                score[:, 0] = float("inf")                       # position 0 (null span) is always kept
                idx = score.topk(min(cfg.evidence_n, ls), dim=-1).indices
                flat = (b_of_q[:, None] * ls + idx).reshape(-1)
                s_sel = S.reshape(-1, d)[flat].view(n, idx.shape[1], d)
                sel_mask = state_mask.reshape(-1)[flat].view(n, idx.shape[1])
                kvs_o = self.stack.project(torch.cat([s_sel, Qn], dim=1))
                mask_o = torch.cat([sel_mask, qmask_ext], dim=1)
            else:
                kq = self.stack.project(Qn)
                kvs_o = [
                    (torch.cat([per_q(ks), qk], dim=2), torch.cat([per_q(vs), qv], dim=2))
                    for (ks, vs), (qk, qv) in zip(kvs_S, kq)
                ]
                mask_o = torch.cat([state_mask_q, qmask_ext], dim=1)
            On = self.stack(Oin, o_mask_flat, rope_o, kvs_o, mask_o, k)

            q_vec = self.q_pool(Qn, qmask_ext)
            u = self.o_pool(On, o_mask_flat).view(n, k, d) + rank
            q2, null2, u2 = self.set_layer(q_vec, u, opt_valid)
            On = On + (u2 - u).reshape(n * k, 1, d)
            a = self.alpha.to(Qn.dtype)
            Qnext = Q + a * (Qn - Q)
            Onext = O + a * (On - O)
            out = readout(q2, u2, null2, opt_valid, null_delta=null_delta) if want_readout else None
            return Qnext, Onext, out, q2, mass

        Q, O = q0, o0
        loops_run = torch.full((n,), plan.n_loops, dtype=torch.long, device=dev)
        per_loop: List[ReadoutOut] = []
        halts: List[torch.Tensor] = []

        if plan.train and torch.is_grad_enabled():
            # Truncated backprop: the first R - k loops run without gradient, the last k carry it.
            free = max(0, plan.n_loops - plan.k_bptt)
            with torch.no_grad():
                for _ in range(free):
                    Q, O, _, _, _ = one_loop(Q, O, False)
            Q, O = Q.detach(), O.detach()
            q_vec = mass = None
            for r in range(free, plan.n_loops):
                Q, O, out, q_vec, mass = one_loop(Q, O, True, want_mass=(r == plan.n_loops - 1))
                per_loop.append(out)
                halts.append(out.halt_logit)
            return {"final": per_loop[-1], "per_loop": per_loop, "halts": halts,
                    "loops_run": loops_run, "q_vec": q_vec, "mass": mass}

        # Inference: stop each question at its halting loop; stop the batch when all have halted.
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        final = q_final = mass_final = None
        for r in range(1, plan.n_loops + 1):
            last = r == plan.n_loops
            Q, O, out, q_vec, mass = one_loop(Q, O, True, want_mass=last or plan.halting)
            halts.append(out.halt_logit)
            if final is None:
                final, q_final, mass_final = out, q_vec, mass
            else:
                keep = ~done          # rows that already halted stay at their halting-loop output
                final = out.where(keep, final)
                q_final = torch.where(keep[:, None], q_vec, q_final)
                if mass is not None:
                    mass_final = mass if mass_final is None else torch.where(keep[:, None], mass, mass_final)
            if plan.halting and not last:
                newly = (torch.sigmoid(out.halt_logit) >= plan.halt_threshold) & (r >= plan.min_loops) & ~done
                loops_run = torch.where(newly, torch.full_like(loops_run, r), loops_run)
                done = done | newly
                if bool(done.all()):
                    break
        return {"final": final, "per_loop": [final], "halts": halts, "loops_run": loops_run,
                "q_vec": q_final, "mass": mass_final}

    @staticmethod
    def _mass_shared(attn: Attention, Qn, kv, state_mask, qmask_ext, m: int) -> torch.Tensor:
        """Attention mass of every question on the shared state, head by head to bound memory."""
        n, lq, d = Qn.shape
        g = n // m
        k, _ = kv
        with torch.no_grad():
            q = attn._project_q(Qn.reshape(g, m * lq, d))
            total = torch.zeros(g, m * lq, k.shape[2], device=Qn.device)
            for h in range(attn.heads):
                sc = torch.matmul(q[:, h].float(), k[:, h].float().transpose(-1, -2)) / math.sqrt(attn.hd)
                sc = sc.masked_fill(~state_mask[:, None, :], float("-inf"))
                total += sc.softmax(dim=-1) / attn.heads
            total = total.view(g, m, lq, -1)
            w = qmask_ext.reshape(g, m, lq, 1).float()
            return ((total * w).sum(dim=2) / w.sum(dim=2).clamp_min(1e-6)).reshape(n, -1)
