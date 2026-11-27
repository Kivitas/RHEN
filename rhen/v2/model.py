"""RHEN-2 model: state encoder -> deliberation workspace -> typed readouts.

The model holds no per-call state, so concurrent calls are safe. ``encode_state`` and
``deliberate`` are separate so a caller can cache the state memory and ask more questions of
the same state without re-encoding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from rhen.v2.config import QUANTILE_LEVELS, TYPE_INDEX, Rhen2Config
from rhen.v2.layers import FactorisedEmbedding, NumericChannel
from rhen.v2.packing import Rhen2Batch
from rhen.v2.readouts import Readout, ReadoutOut
from rhen.v2.state_encoder import StateEncoder
from rhen.v2.workspace import DeliberationWorkspace, Interventions, LoopPlan

SCORE = TYPE_INDEX["score"]


@dataclass
class Rhen2Output:
    final: ReadoutOut                 # last (or halting) loop, rows = B * M questions
    per_loop: List[ReadoutOut]        # gradient-carrying loops in training, [final] in inference
    halts: List[torch.Tensor]
    loops_run: torch.Tensor           # [N]
    probs: torch.Tensor               # [N, K+1] resolved per type (choice/noul softmax, score link), null last
    multi_prob: torch.Tensor          # [N, K]
    quantiles: torch.Tensor           # [N, 9] scaled into each question's [lo, hi]
    span_start: torch.Tensor          # [N, Ls] logits
    span_end: torch.Tensor            # [N, Ls] logits
    evidence: Optional[torch.Tensor]  # [N, Ls] attention mass on the state, sums to 1
    q_vec: torch.Tensor               # [N, d]
    batch_size: int
    max_questions: int
    loss: Optional[torch.Tensor] = None
    loss_parts: Dict[str, float] = field(default_factory=dict)


class Rhen2Model(nn.Module):
    def __init__(self, cfg: Rhen2Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = FactorisedEmbedding(cfg.vocab_size, cfg.d_embed, d)
        self.segment = nn.Embedding(3, d)               # state, question, option
        self.qtype = nn.Embedding(len(TYPE_INDEX), d)
        self.numeric = NumericChannel(d) if cfg.use_numeric else None
        self.encoder = StateEncoder(cfg)
        self.workspace = DeliberationWorkspace(cfg)
        self.readout = Readout(cfg)
        self.mask_memory = nn.Parameter(torch.zeros(d))  # stands in for deleted evidence (Checker C3a)
        self.apply(self._init)
        scale = (2 * (cfg.n_state_layers + cfg.n_core_layers)) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("ffn.down.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std * scale)
        with torch.no_grad():
            self.workspace.rank_emb.normal_(0.0, cfg.init_std)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- encoding ------------------------------------------------------------
    def encode_state(self, state_ids: torch.Tensor, state_mask: torch.Tensor, state_num: torch.Tensor) -> torch.Tensor:
        x = self.embed(state_ids) + self.segment.weight[0]
        if self.numeric is not None:
            x = x + self.numeric(state_num.to(x.dtype))
        return self.encoder(x, state_mask)

    def _question_inputs(self, batch: Rhen2Batch):
        b, m, lq = batch.q_ids.shape
        k, lo = batch.opt_ids.shape[2], batch.opt_ids.shape[3]
        n = b * m
        q = self.embed(batch.q_ids) + self.segment.weight[1] + self.qtype(batch.q_type)[:, :, None]
        if self.numeric is not None:
            q = q + self.numeric(batch.q_num.to(q.dtype))
        o = self.embed(batch.opt_ids) + self.segment.weight[2]
        return (
            q.reshape(n, lq, -1), batch.q_mask.reshape(n, lq),
            o.reshape(n, k, lo, -1), batch.opt_mask.reshape(n, k, lo),
            batch.opt_valid.reshape(n, k), batch.q_type.reshape(n),
        )

    # -- deliberation ----------------------------------------------------------
    def deliberate(
        self,
        S: torch.Tensor,
        batch: Rhen2Batch,
        loops: LoopPlan,
        interventions: Optional[Interventions] = None,
    ) -> Rhen2Output:
        b, m = batch.q_ids.shape[:2]
        n = b * m
        q_emb, q_mask, o_emb, o_mask, opt_valid, q_type = self._question_inputs(batch)
        state_mask = batch.state_mask
        per_question = False
        null_delta = 0.0
        mem = S
        if interventions is not None:
            null_delta = float(interventions.null_delta)
            if interventions.memory_mask is not None or interventions.empty_memory:
                per_question = True
                b_of_q = torch.arange(n, device=S.device) // m
                mem = S[b_of_q]
                state_mask = batch.state_mask[b_of_q]
                if interventions.memory_mask is not None:
                    mem = torch.where(interventions.memory_mask[:, :, None], self.mask_memory.to(mem.dtype), mem)
                if interventions.empty_memory:
                    state_mask = torch.zeros_like(state_mask)
                    state_mask[:, 0] = True
        ws = self.workspace(
            mem, state_mask, q_emb, q_mask, o_emb, o_mask, opt_valid, q_type,
            per_question=per_question, questions_per_example=m, plan=loops,
            readout=self.readout, null_delta=null_delta,
        )
        final: ReadoutOut = ws["final"]
        q_vec = ws["q_vec"]

        s_proj = self.readout.project_state_for_spans(mem)
        start, end = self.readout.span_logits(q_vec, s_proj, state_mask, group=1 if per_question else m)

        is_score = (q_type == SCORE)[:, None]
        probs = torch.where(is_score, final.score_probs, final.probs)
        lo = batch.range_lo.reshape(n, 1)
        hi = batch.range_hi.reshape(n, 1)
        evidence = ws["mass"]
        if evidence is not None:
            evidence = evidence / evidence.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return Rhen2Output(
            final=final, per_loop=ws["per_loop"], halts=ws["halts"], loops_run=ws["loops_run"],
            probs=probs, multi_prob=torch.sigmoid(final.multi_logit),
            quantiles=lo + (hi - lo) * final.quantiles,
            span_start=start, span_end=end, evidence=evidence, q_vec=q_vec,
            batch_size=b, max_questions=m,
        )

    def forward(self, batch: Rhen2Batch, loops: Optional[LoopPlan] = None) -> Rhen2Output:
        if loops is None:
            loops = LoopPlan.sample(self.cfg) if self.training else LoopPlan.effort(self.cfg, "medium")
        S = self.encode_state(batch.state_ids, batch.state_mask, batch.state_num)
        out = self.deliberate(S, batch, loops)
        if batch.label_idx is not None:
            from rhen.v2.losses import compute_losses

            out.loss, parts = compute_losses(out, batch, self.cfg)
            out.loss_parts = parts
        return out


__all__ = ["Rhen2Model", "Rhen2Output", "QUANTILE_LEVELS"]
