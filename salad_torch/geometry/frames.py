"""Backbone frames and idealised-geometry helpers."""
from __future__ import annotations

import torch
from torch import Tensor

from .rigid import Rigid, vec_cross, vec_normalized

__all__ = ["extract_aa_frames", "make_transform_from_reference",
           "compute_pseudo_cb", "positions_to_ncacocb"]

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
    local = frames[..., None].apply_inverse_to_point(pos)
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
