"""Segment ("index_*") reductions over the flat residue axis.

salad packs a whole batch into one flat residue axis `N` with companion
`batch_index` / `chain_index` vectors; every "global" reduction is a segment op.

All of these follow the same shape contract as the JAX originals: the result has
**the same leading size as the input**, not the number of segments. The reduced
value is broadcast back to every member of its segment.

`mask` is expected to broadcast against `data`, i.e. callers pass
``mask[..., None]`` when `data` has a trailing feature axis - same as upstream.
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["index_sum", "index_mean", "index_max", "index_var", "index_std",
           "index_count", "unique_chain"]


def _segments(index: Tensor) -> int:
    return int(index.max().item()) + 1 if index.numel() else 0


def index_sum(data: Tensor, index: Tensor, mask: Tensor, apply_mask: bool = True) -> Tensor:
    """Per-segment sum, broadcast back. Mirrors `geometry.index_sum`."""
    masked = torch.where(mask.bool(), data, torch.zeros_like(data))
    out = torch.zeros_like(data)
    idx = index.view(-1, *([1] * (data.dim() - 1))).expand_as(data)
    out = out.scatter_add(0, idx, masked)
    result = out[index]
    if apply_mask:
        result = torch.where(mask.bool(), result, torch.zeros_like(result))
    return result


def index_count(index: Tensor, mask: Tensor, apply_mask: bool = True) -> Tensor:
    """Per-segment count of unmasked members, broadcast back.

    Accumulates in the *index* dtype (integer), as upstream does.
    """
    out = torch.zeros_like(index)
    out = out.scatter_add(0, index, mask.to(index.dtype))
    result = out[index]
    if apply_mask:
        return torch.where(mask.bool(), result, torch.zeros_like(result))
    return result


def index_mean(data: Tensor, index: Tensor, mask: Tensor,
               weight: Tensor | None = None, apply_mask: bool = True) -> Tensor:
    """Per-segment (optionally weighted) mean, broadcast back.

    Note the 1e-6 floor on the denominator, copied from upstream.
    """
    mask_b = mask.bool()
    if weight is None:
        weight = torch.ones_like(data)
    else:
        weight = weight.expand_as(data)
    # upstream mutates `data *= weight`; we must not touch the caller's tensor
    weighted = data * weight
    total = index_sum(weighted, index, mask_b, apply_mask=False)
    count = index_sum(weight, index, mask_b, apply_mask=False)
    result = total / count.clamp(min=1e-6)
    if apply_mask:
        result = torch.where(mask_b, result, torch.zeros_like(result))
    return result


def index_max(data: Tensor, index: Tensor, mask: Tensor, apply_mask: bool = True) -> Tensor:
    """Per-segment max, broadcast back.

    Upstream seeds the accumulator with ``data.min()`` (a global scalar), so
    fully-masked segments come out as that scalar. Reproduced exactly.
    """
    floor = data.min()
    masked = torch.where(mask.bool(), data, torch.full_like(data, float(floor)))
    out = torch.full_like(data, float(floor))
    idx = index.view(-1, *([1] * (data.dim() - 1))).expand_as(data)
    out = out.scatter_reduce(0, idx, masked, reduce="amax", include_self=True)
    result = out[index]
    if apply_mask:
        # masked-out entries fall back to the global minimum, not to zero
        result = torch.where(mask.bool(), result, torch.full_like(result, float(floor)))
    return result


def index_var(data: Tensor, index: Tensor, mask: Tensor, apply_mask: bool = True) -> Tensor:
    mean = index_mean(data, index, mask, apply_mask=False)
    var = index_mean((data - mean) ** 2, index, mask, apply_mask=False)
    if apply_mask:
        var = torch.where(mask.bool(), var, torch.zeros_like(var))
    return var


def index_std(data: Tensor, index: Tensor, mask: Tensor, apply_mask: bool = True,
              eps: float = 1e-6) -> Tensor:
    """``sqrt(index_var(..., apply_mask) + eps)``.

    The epsilon is added *after* masking, so masked entries come out as
    ``sqrt(eps)`` (1e-3), not 0. That asymmetry is upstream behaviour.
    """
    return torch.sqrt(index_var(data, index, mask, apply_mask) + eps)


def unique_chain(chain: Tensor, batch: Tensor) -> Tensor:
    """Globally unique chain ids across a flat multi-protein batch.

    Upstream uses `lax.scan`; the same result is a cumulative sum of boundary
    flags, which is both faster and differentiable-free here.
    """
    if chain.numel() == 0:
        return chain
    boundary = torch.zeros_like(chain, dtype=torch.long)
    changed = (chain[1:] != chain[:-1]) | (batch[1:] != batch[:-1])
    boundary[1:] = changed.long()
    return torch.cumsum(boundary, dim=0).to(chain.dtype)
