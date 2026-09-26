"""Exponential moving average of the model's trainable parameters.

Port of `ema_step` (spec §B.4): decay `0.999` for `default_vp`, and the
released `checkpoint-ema-*` files - not the raw `checkpoint-*` ones - are what
the paper's eval/sampling results actually came from. Only
`nn.Module.parameters()` are tracked, never buffers: Haiku's `params` pytree
that `ema_step` maps over corresponds 1:1 to `.parameters()`, not
`.state_dict()` (which would additionally sweep up any non-trainable buffers,
if this model had any).

**Off-by-one staleness (spec §B.4), reproduced by construction, not tuned
away**: upstream calls `ema_step(loop_state.params, ...)` with the parameters
AS THEY WERE BEFORE that step's `update()` runs, so the EMA saved at step `N`
is `decay*ema_{N-1} + (1-decay)*params_{N-1}` - one full gradient step stale
relative to a raw checkpoint saved at the same step id. `EMA.update()`
reproduces this automatically as long as the CALLER invokes it before
`optimizer.step()` (see `training/loop.py`'s `training_step`); calling it
after would be a deliberate, documented deviation (more conventional
semantics, not bit-exact against a JAX run at any finite step).
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["EMA"]


class EMA:
    """Holds one EMA "shadow" tensor per named parameter of `model`."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the shadow towards `model`'s CURRENT parameter values.

        Blending an EMA that was just initialised from `model`'s own params
        (the "first call" case, `ema_step`'s `"ema_params" not in aux_state`
        branch) is a no-op here by construction - `decay*x + (1-decay)*x == x`
        - so no separate first-call special case is needed.
        """
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for key, value in state.items():
            self.shadow[key] = value.clone()

    def copy_to(self, model: nn.Module) -> None:
        """Overwrite `model`'s parameters in place with the EMA shadow - for
        loading a `checkpoint-ema-*` snapshot into a model for eval/sampling."""
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(self.shadow[name])
