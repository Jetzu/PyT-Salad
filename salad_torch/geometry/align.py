"""Kabsch/Umeyama alignment over the flat multi-item residue axis.

Ported from `salad/modules/utils/geometry.py:647-701`
(`notes/specs/03_training_losses.md` §4). **This is DEAD for `default_vp`**:
`c.kabsch` is unset, so `dotdict.__getattr__ = dict.get` makes it `None` ->
falsy, and every position/atom/FAPE loss in `losses.structure` compares raw,
un-aligned world-frame coordinates. It is implemented anyway because other
configs (`semi_equivariant_kabsch_vp`, `default_vp_atom14_kabsch`) turn it on,
and callers should gate it behind their own `config.kabsch` check rather than
this module silently doing nothing.

Convention: `index_align(x, y, ...)` moves `x` onto `y` - the FIRST argument is
the one that gets transformed, the SECOND is the fixed target. Where salad
uses this (guarded by `if c.kabsch:`), it aligns `pos_gt` onto `pos_pred` -
the opposite of the usual "align prediction to target" convention. If that
flag is ever turned on, the loss's "ground truth" becomes a moving target
that chases the prediction's current global pose.

Gradient flow through `index_kabsch`, precisely: only the `covariance` tensor
fed to `torch.linalg.svd` is detached (`jax.lax.stop_gradient` in the
original). Consequences:
  * `rot` carries ZERO gradient to either input - the optimal rotation is
    treated as a constant, even though it was computed from both `x` and `y`.
  * `x_center`/`y_center` are NOT detached - gradient flows normally through
    the per-item mean subtraction.
  * In `index_align`'s result, `y` (typically `pos_pred`) receives gradient
    ONLY through the additive `y_center` translation term (no rotational
    coupling); `x` (typically `pos_gt`) receives gradient through both
    `x_center` and the `(x - x_center)` term rotated by the (frozen) `rot`.
  * Net effect if ever enabled: this loss path cannot teach the model
    anything about rotational agreement - only about translation. Any
    rotational learning signal has to come from elsewhere (the separate
    `rotation` loss, `losses.structure.rotation_loss`).
"""
from __future__ import annotations

import torch
from torch import Tensor

from .segment import index_mean

__all__ = ["index_kabsch", "index_align", "apply_alignment"]


def index_kabsch(x: Tensor, y: Tensor, index: Tensor, mask: Tensor,
                 weight: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Per-item optimal rotation aligning `x` onto `y`.

    Args:
        x, y: `(L, 3)` points (already atom-sliced by the caller - `index_align`
            passes only the CA slot, never a full multi-atom tensor).
        index: `(L,)` segment id (batch_index).
        mask: `(L,)` validity.
        weight: optional `(L, 1)` per-residue weight; defaults to all-ones.
    Returns:
        `(rot, x_center, y_center)`, each broadcast back to `(L, 3, 3)` /
        `(L, 3)` per the house `index_*` convention (segment.py) - every
        residue of a given item holds that item's shared value.
    """
    if weight is not None and weight.dim() == 1:
        # JAX takes a (L,) weight and unsqueezes internally; accept either so a
        # caller copying an upstream call site does not trip.
        weight = weight[:, None]
    x_center = index_mean(x, index, mask[:, None], weight)
    y_center = index_mean(y, index, mask[:, None], weight)
    x = x - x_center
    y = y - y_center
    w = weight if weight is not None else torch.ones_like(index, dtype=x.dtype)[:, None]

    from .segment import index_sum  # local import: avoids a module-level cycle risk
    covariance = index_sum(w[:, :, None] * x[:, :, None] * y[:, None, :], index,
                           mask[:, None, None])

    # SVD input is DETACHED - see module docstring for the exact gradient
    # consequences. `torch.linalg.svd(..., full_matrices=True)` returns
    # `(U, S, Vh)`, the same convention as JAX's `svd(compute_uv=True)`, so
    # the JAX code's `v` below is already `Vh` and this ports with no
    # transpose fix-up.
    u, _, v = torch.linalg.svd(covariance.detach(), full_matrices=True)
    det_sign = torch.linalg.det(u) * torch.linalg.det(v)
    det_flip = torch.ones_like(u[..., 0, :])           # (L, 3)
    det_flip = det_flip.clone()
    det_flip[..., -1] = det_sign
    rot = torch.einsum("...ak,...kb->...ba", u * det_flip[..., None, :], v)
    return rot, x_center, y_center


def apply_alignment(x: Tensor, rot: Tensor, x_center: Tensor, y_center: Tensor) -> Tensor:
    """Apply an already-computed Kabsch transform to any `(L, A, 3)` tensor.

    Factored out of `index_align` so a caller holding one `(rot, x_center,
    y_center)` triple (e.g. from aligning `pos`) can reuse it on a second
    tensor (e.g. `atom_pos`) without recomputing the SVD. Not called by
    `index_align` itself, and not exercised by `default_vp` (§0(a): kabsch is
    off) - provided for configs that align multiple tensors under one shared
    transform.
    """
    return torch.einsum("...ak,...ik->...ia", rot, x - x_center[:, None]) + y_center[:, None]


def index_align(x: Tensor, y: Tensor, index: Tensor, mask: Tensor,
                weight: Tensor | None = None) -> Tensor:
    """Rigidly transform `x` onto `y`; `x` moves, `y` is the fixed target.

    Alignment is determined from the CA slot only (`x[:, 1]`, `y[:, 1]`) but
    applied to every atom of `x` - matching `index_kabsch(x[:, 1], y[:, 1], ...)`
    then broadcasting the resulting rotation over the atom axis.
    """
    rot, x_center, y_center = index_kabsch(x[:, 1], y[:, 1], index, mask, weight)
    return apply_alignment(x, rot, x_center, y_center)
