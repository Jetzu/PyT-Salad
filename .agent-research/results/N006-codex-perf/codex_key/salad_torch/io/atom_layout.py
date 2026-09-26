"""Conversions between the atom14 and atom37 heavy-atom layouts.

Port of the four layout helpers in `salad.aflib.model.all_atom_multimer`
(:207-241). The model works in atom14 (``N, CA, C, O, CB, ...``); PDB output
wants atom37 (``N, CA, C, CB, O, CG, ...``), so the writer has to cross over.

Both index maps use 0 as the "atom not present" sentinel, so a bare gather
duplicates atom N into every absent slot. The mask multiply that follows is
therefore load-bearing, not defensive - dropping it writes a pile of N atoms at
every sidechain position.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .residue_constants import (RESTYPE_ATOM14_MASK, RESTYPE_ATOM14_TO_ATOM37,
                                RESTYPE_ATOM37_MASK, RESTYPE_ATOM37_TO_ATOM14,
                                table_to)

__all__ = ["atom14_to_atom37", "atom37_to_atom14", "get_atom37_mask",
           "get_atom14_mask"]


def _lookup(table: Tensor, aatype: Tensor) -> Tensor:
    """``table[aatype]`` with the clip that `batched_gather(mode='clip')` applies."""
    idx = aatype.long().clamp(0, table.shape[0] - 1)
    return table_to(table, aatype.device)[idx]


def get_atom37_mask(aatype: Tensor) -> Tensor:
    """``(N,) -> (N, 37)`` float32. ``all_atom_multimer.py:207-208``.

    Row 20 (UNK) is all zeros, so an unknown residue contributes no atoms.
    """
    return _lookup(RESTYPE_ATOM37_MASK, aatype)


def get_atom14_mask(aatype: Tensor) -> Tensor:
    """``(N,) -> (N, 14)`` float32."""
    return _lookup(RESTYPE_ATOM14_MASK, aatype)


def atom14_to_atom37(atom14_data: Tensor, aatype: Tensor) -> Tensor:
    """``(N, 14[, C]) -> (N, 37[, C])``. ``all_atom_multimer.py:228-241``."""
    if atom14_data.dim() not in (2, 3):
        raise ValueError(
            f"atom14_data must be (N, 14) or (N, 14, C), got {tuple(atom14_data.shape)}")
    idx = _lookup(RESTYPE_ATOM37_TO_ATOM14, aatype).long().clamp(0, 13)  # (N, 37)
    mask37 = _lookup(RESTYPE_ATOM37_MASK, aatype).to(atom14_data.dtype)
    if atom14_data.dim() == 2:
        return torch.gather(atom14_data, 1, idx) * mask37
    channels = atom14_data.shape[-1]
    gathered = torch.gather(atom14_data, 1, idx[..., None].expand(-1, -1, channels))
    return gathered * mask37[:, :, None]


def atom37_to_atom14(aatype: Tensor, atom37_data: Tensor,
                     atom37_mask: Tensor) -> tuple[Tensor, Tensor]:
    """``(N, 37[, C]) -> ((N, 14[, C]), (N, 14))``, the inverse gather.

    The returned mask is the residue's atom14 mask *and* the supplied atom37
    mask, so atoms that are chemically possible but missing from the input stay
    masked out.
    """
    idx = _lookup(RESTYPE_ATOM14_TO_ATOM37, aatype).long().clamp(0, 36)  # (N, 14)
    atom14_mask = (_lookup(RESTYPE_ATOM14_MASK, aatype)
                   * torch.gather(atom37_mask.to(torch.float32), 1, idx))
    if atom37_data.dim() == 2:
        atom14 = torch.gather(atom37_data, 1, idx) * atom14_mask.to(atom37_data.dtype)
    else:
        channels = atom37_data.shape[-1]
        atom14 = torch.gather(atom37_data, 1, idx[..., None].expand(-1, -1, channels))
        atom14 = atom14 * atom14_mask.to(atom37_data.dtype)[:, :, None]
    return atom14, atom14_mask
