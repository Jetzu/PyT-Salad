"""Backbone frames and idealised-geometry helpers."""
from __future__ import annotations

import torch
from torch import Tensor

from .. import opt
from .rigid import Rigid, rot_from_two_vectors, vec_cross, vec_normalized

__all__ = ["extract_aa_frames", "make_transform_from_reference",
           "compute_pseudo_cb", "positions_to_ncacocb",
           "aa_frames_t", "apply_t", "apply_inverse_t"]

# idealised CB placement constants, from `geometry.compute_pseudo_cb`
_CB_CONST = (-0.58273431, 0.56802827, -0.54067466)


def make_transform_from_reference(a: Tensor, b: Tensor, c: Tensor) -> Rigid:
    """Frame with origin at `b`, x-axis along `c - b`, `a` in the +xy plane.

    Called as ``(N, CA, C)``, so the x-axis runs CA->C and N sets the xy plane.
    """
    return Rigid.from_reference(a, b, c)


def extract_aa_frames(pos: Tensor) -> tuple[Rigid, Tensor]:
    """Per-residue backbone frames and frame-local atom coordinates.

    Args:
        pos: ``(..., A, 3)`` atom positions, atom order N, CA, C, ...
    Returns:
        ``(frames, local_pos)`` where frames has shape ``(...)`` and local_pos
        is ``(..., A, 3)``.
    """
    frames = make_transform_from_reference(pos[..., 0, :], pos[..., 1, :], pos[..., 2, :])
    local = frames.unsqueeze_batch(1).apply_inverse_to_point(pos)
    return frames, local


def compute_pseudo_cb(positions: Tensor) -> Tensor:
    """Idealised CB from N, CA, C. Input ``(..., >=3, 3)``, output ``(..., 3)``."""
    n = positions[..., 0, :]
    ca = positions[..., 1, :]
    c = positions[..., 2, :]
    b = ca - n
    cc = c - ca
    a = vec_cross(b, cc)
    return _CB_CONST[0] * a + _CB_CONST[1] * b + _CB_CONST[2] * cc + ca


def positions_to_ncacocb(positions: Tensor) -> Tensor:
    """Take ``(..., >=4, 3)`` N/CA/C/O and append an idealised CB -> ``(..., 5, 3)``."""
    cb = compute_pseudo_cb(positions)
    return torch.cat([positions[..., :4, :], cb.unsqueeze(-2)], dim=-2)


# --------------------------------------------------------------------------
# Tensor-level equivalents, for the torch.compile hot path.
#
# `Rigid` is a plain Python object, which Dynamo treats as opaque: it breaks
# the graph on every operator overload ("builtin: matmul
# [UserDefinedObjectVariable, ...]"). Graph breaks disable CUDA graphs for the
# whole region, and this model is launch-bound, so that costs several x.
# These functions do the identical arithmetic on bare tensors so the compiled
# region never constructs a Rigid. The class stays for the eager API, where
# readability matters more than fusion.
# --------------------------------------------------------------------------
def aa_frames_t(pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """`extract_aa_frames` without the wrapper: ``(rot, trans, local_pos)``."""
    n, ca, c = pos[..., 0, :], pos[..., 1, :], pos[..., 2, :]
    rot = rot_from_two_vectors(c - ca, n - ca)
    local = apply_inverse_t(rot, ca, pos, k=1)
    return rot, ca, local


def _lift(rot: Tensor, trans: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Insert `k` axes after the batch dims and before the trailing ones."""
    for _ in range(k):
        rot = rot.unsqueeze(-3)
        trans = trans.unsqueeze(-2)
    return rot, trans


def apply_t(rot: Tensor, trans: Tensor, points: Tensor, k: int = 0) -> Tensor:
    """``R p + t``, broadcasting over `k` inserted axes."""
    rot, trans = _lift(rot, trans, k)
    if opt.FRAME_MULSUM:
        # O8 - a 3x3 matvec as a fusable reduction, not a cuBLAS bmm.
        return (rot * points.unsqueeze(-2)).sum(-1) + trans
    return torch.einsum("...ij,...j->...i", rot, points) + trans


def apply_inverse_t(rot: Tensor, trans: Tensor, points: Tensor, k: int = 0) -> Tensor:
    """``R^T (p - t)``, broadcasting over `k` inserted axes."""
    rot, trans = _lift(rot, trans, k)
    if opt.FRAME_MULSUM:
        return (rot * (points - trans).unsqueeze(-1)).sum(-2)
    return torch.einsum("...ji,...j->...i", rot, points - trans)
