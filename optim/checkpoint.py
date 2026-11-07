"""Gradient checkpointing for the recurrent core.

``CheckpointedRecurrentCore`` wraps the recurrent core and recomputes its
forward pass during backward (non-reentrant ``torch.utils.checkpoint``). The
recomputation runs only in training mode and only when checkpointing is enabled.

The core carries per-sequence state across loop steps (the SSM highway memory and,
when enabled, the Titans memory matrix). Backward recomputes the loop steps in
reverse order, after the forward has left that state at its final value, so the
wrapper snapshots the state each step starts from and restores it before the
recomputation. The MoE congestion bias is the other piece of mutable state; use
the ``"step"`` balance update mode with checkpointing (see ``SharedExpertFineGrainedMoE``).
"""

# (module attribute path, state attribute) pairs carried across loop steps.
_LOOP_STATE_ATTRS = (("ssm_highway", "memory_state"), ("titans_memory", "memory_matrix"))

import torch
import torch.nn as nn
import torch.utils.checkpoint


class CheckpointedRecurrentCore(nn.Module):
    """Recurrent core with whole-core gradient checkpointing.

    Args:
        core: The module to wrap.
        config: Optional config. When it has a ``gradient_checkpointing``
            attribute, that value takes precedence over the argument below.
        gradient_checkpointing: Used when ``config`` is None or lacks the attribute.
    """

    def __init__(self, core: nn.Module, config=None, gradient_checkpointing: bool = True):
        super().__init__()
        self.core = core
        if config is not None and hasattr(config, "gradient_checkpointing"):
            gradient_checkpointing = config.gradient_checkpointing
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def _snapshot_loop_state(self):
        snapshot = []
        for module_name, attr in _LOOP_STATE_ATTRS:
            module = getattr(self.core, module_name, None)
            if module is not None:
                snapshot.append((module, attr, getattr(module, attr, None)))
        return snapshot

    @staticmethod
    def _restore_loop_state(snapshot) -> None:
        for module, attr, value in snapshot:
            setattr(module, attr, value)

    def forward(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            snapshot = self._snapshot_loop_state()

            def run_core(*inputs):
                # A no-op in the first (forward) call; in the backward recomputation it
                # rewinds the loop state to what this step originally started from.
                self._restore_loop_state(snapshot)
                return self.core(*inputs, **kwargs)
            # RNG state is preserved (the torch default), so dropout masks that are
            # recomputed during backward match the ones drawn in the forward pass.
            return torch.utils.checkpoint.checkpoint(run_core, *args, use_reentrant=False)
        return self.core(*args, **kwargs)

    def reset_special_parameters(self):
        """Delegate to the wrapped core."""
        if hasattr(self.core, "reset_special_parameters"):
            self.core.reset_special_parameters()