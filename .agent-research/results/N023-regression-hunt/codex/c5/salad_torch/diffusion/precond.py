"""Input/output preconditioning scale factors (VP and VE/EDM)."""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["edm_scaling", "preconditioning_scale_factors"]


def _as_tensor(x, like: Tensor | None = None) -> Tensor:
    if torch.is_tensor(x):
        return x
    ref = like if like is not None else torch.tensor(0.0)
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def edm_scaling(sigma_data, sigma_noise, alpha=1.0, beta=1.0) -> dict:
    """EDM in/out/skip/loss scales. Both sigmas are floored at 1e-3."""
    ref = sigma_noise if torch.is_tensor(sigma_noise) else sigma_data
    sigma_data = _as_tensor(sigma_data, ref if torch.is_tensor(ref) else None)
    sigma_noise = _as_tensor(sigma_noise, sigma_data)
    beta = _as_tensor(beta, sigma_data)
    alpha = _as_tensor(alpha, sigma_data)

    sigma_data = torch.clamp(sigma_data, min=1e-3)
    sigma_noise = torch.clamp(sigma_noise, min=1e-3)
    denominator = alpha ** 2 * sigma_data ** 2 + beta ** 2 * sigma_noise ** 2
    in_scale = 1 / torch.sqrt(denominator)
    out_scale = beta * sigma_data * sigma_noise / torch.sqrt(denominator)
    skip_scale = alpha * sigma_data ** 2 / denominator
    return {
        "in": in_scale,
        "out": out_scale,
        "skip": skip_scale,
        "refine": 1 / sigma_data,
        "loss": 1 / out_scale ** 2,
    }


def preconditioning_scale_factors(config, t, sigma_data) -> dict:
    """Dispatch on ``config.preconditioning``.

    For plain VP this is the constant ``{in:1, out:sigma_data, skip:1, loss:1}``,
    i.e. every block's position update is scaled by a flat 10 A.
    """
    c = config
    if c.preconditioning in ("edm", "edm_scaled"):
        result = edm_scaling(sigma_data, 1.0, alpha=1.0, beta=t)
    elif c.preconditioning == "flow":
        t_t = _as_tensor(t)
        result = {"in": 1.0, "out": sigma_data, "skip": 1.0, "refine": 1,
                  "loss": 1 / torch.clamp(t_t ** 2, min=1e-6)}
    else:
        result = {"in": 1.0, "out": sigma_data, "skip": 1.0, "refine": 1, "loss": 1}
    if not c.refine:
        result["refine"] = 1.0
    return result
