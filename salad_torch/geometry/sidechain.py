"""Sidechain construction: torsion angles -> atom14 coordinates.

This is AlphaFold's rigid-group machinery, as vendored by salad in
``salad/aflib/model/all_atom_multimer.py``. A residue is a tree of 8 rigid
groups (backbone, pre-omega, phi, psi, chi1..chi4); each group rotates about the
x-axis of its parent by one torsion angle, and every heavy atom sits at a fixed
literal position inside exactly one group. ``single_protein_sidechains`` is the
only entry point the model uses - the `AngleHead` predicts 7 (sin, cos) pairs
and this turns them into coordinates.

No learned parameters live here, only the four AlphaFold constant tables.

Constant tables
---------------
Rather than vendor the whole 1100-line ``residue_constants`` module (which pulls
in ``dm-tree``), the four tables this port needs are rebuilt at import from
`_ATOM14`, the atom14-ordered form of AlphaFold's ``rigid_group_atom_positions``.
`restype_rigid_group_default_frame` is then derived exactly as
``_make_rigid_group_constants`` does, using two regularities that hold for all
20 canonical residues (checked against the reference table, which the result
reproduces *bitwise*):

  * the chi1 frame is always built from (N, CA, CB), i.e. atom14 indices 0, 1, 4
  * the chi(k+1) rotation axis always ends at atom14 index ``4 + k``
    (CB, CG, CD, NE/CE), so no chi-atom name table is needed

Index 20 is UNK. AlphaFold's builder loops over the 20 canonical types only, so
row 20 of *every* table - including the frames, whose group 0 is the identity
everywhere else - stays all zeros. Reproduced: a residue typed 20 yields zero
sidechain coordinates and a zero atom mask, which is what the reference does.

Conventions: rotations are row-major ``(..., 3, 3)``, matching both salad's JAX
`Rot3Array` and `salad_torch.geometry.rigid`, so nothing here transposes.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .rigid import Rigid

__all__ = [
    "torsion_angles_to_frames",
    "frames_and_literature_positions_to_atom14_pos",
    "single_protein_sidechains",
    "restype_rigid_group_default_frame",
    "restype_atom14_to_rigid_group",
    "restype_atom14_rigid_group_positions",
    "restype_atom14_mask",
    "ATOM14_NAMES",
    "RESTYPE_3",
]

# AlphaFold restype order (alphabetical by *three*-letter code, not one-letter)
RESTYPE_1 = "ARNDCQEGHILKMFPSTWYV"
RESTYPE_3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

# (atom name, rigid group index, literal position inside that group), in atom14
# order. Verbatim from AlphaFold's `rigid_group_atom_positions` (Apache-2.0,
# DeepMind), reindexed to atom14 and with trailing zeros trimmed.
_ATOM14: dict[str, list[tuple[str, int, tuple[float, float, float]]]] = {
    "ALA": [
        ("N", 0, (-0.525, 1.363, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.526, 0, 0)),
        ("O", 3, (0.627, 1.062, 0)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
    ],
    "ARG": [
        ("N", 0, (-0.524, 1.362, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.525, 0, 0)),
        ("O", 3, (0.626, 1.062, 0)),
        ("CB", 0, (-0.524, -0.778, -1.209)),
        ("CG", 4, (0.616, 1.39, 0)),
        ("CD", 5, (0.564, 1.414, 0)),
        ("NE", 6, (0.539, 1.357, 0)),
        ("CZ", 7, (0.758, 1.093, 0)),
        ("NH1", 7, (0.206, 2.301, 0)),
        ("NH2", 7, (2.078, 0.978, 0)),
    ],
    "ASN": [
        ("N", 0, (-0.536, 1.357, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.526, 0, 0)),
        ("O", 3, (0.625, 1.062, 0)),
        ("CB", 0, (-0.531, -0.787, -1.2)),
        ("CG", 4, (0.584, 1.399, 0)),
        ("OD1", 5, (0.633, 1.059, 0)),
        ("ND2", 5, (0.593, -1.188, 0.001)),
    ],
    "ASP": [
        ("N", 0, (-0.525, 1.362, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.527, 0, 0)),
        ("O", 3, (0.626, 1.062, 0)),
        ("CB", 0, (-0.526, -0.778, -1.208)),
        ("CG", 4, (0.593, 1.398, 0)),
        ("OD1", 5, (0.61, 1.091, 0)),
        ("OD2", 5, (0.592, -1.101, -0.003)),
    ],
    "CYS": [
        ("N", 0, (-0.522, 1.362, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.524, 0, 0)),
        ("O", 3, (0.625, 1.062, 0)),
        ("CB", 0, (-0.519, -0.773, -1.212)),
        ("SG", 4, (0.728, 1.653, 0)),
    ],
    "GLN": [
        ("N", 0, (-0.526, 1.361, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.526, 0, 0)),
        ("O", 3, (0.626, 1.062, 0)),
        ("CB", 0, (-0.525, -0.779, -1.207)),
        ("CG", 4, (0.615, 1.393, 0)),
        ("CD", 5, (0.587, 1.399, 0)),
        ("OE1", 6, (0.634, 1.06, 0)),
        ("NE2", 6, (0.593, -1.189, -0.001)),
    ],
    "GLU": [
        ("N", 0, (-0.528, 1.361, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.526, 0, 0)),
        ("O", 3, (0.626, 1.062, 0)),
        ("CB", 0, (-0.526, -0.781, -1.207)),
        ("CG", 4, (0.615, 1.392, 0)),
        ("CD", 5, (0.6, 1.397, 0)),
        ("OE1", 6, (0.607, 1.095, 0)),
        ("OE2", 6, (0.589, -1.104, -0.001)),
    ],
    "GLY": [
        ("N", 0, (-0.572, 1.337, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.517, 0, 0)),
        ("O", 3, (0.626, 1.062, 0)),
    ],
    "HIS": [
        ("N", 0, (-0.527, 1.36, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.525, 0, 0)),
        ("O", 3, (0.625, 1.063, 0)),
        ("CB", 0, (-0.525, -0.778, -1.208)),
        ("CG", 4, (0.6, 1.37, 0)),
        ("ND1", 5, (0.744, 1.16, 0)),
        ("CD2", 5, (0.889, -1.021, 0.003)),
        ("CE1", 5, (2.03, 0.851, 0.002)),
        ("NE2", 5, (2.145, -0.466, 0.004)),
    ],
    "ILE": [
        ("N", 0, (-0.493, 1.373, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.527, 0, 0)),
        ("O", 3, (0.627, 1.062, 0)),
        ("CB", 0, (-0.536, -0.793, -1.213)),
        ("CG1", 4, (0.534, 1.437, 0)),
        ("CG2", 4, (0.54, -0.785, -1.199)),
        ("CD1", 5, (0.619, 1.391, 0)),
    ],
    "LEU": [
        ("N", 0, (-0.52, 1.363, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.525, 0, 0)),
        ("O", 3, (0.625, 1.063, 0)),
        ("CB", 0, (-0.522, -0.773, -1.214)),
        ("CG", 4, (0.678, 1.371, 0)),
        ("CD1", 5, (0.53, 1.43, 0)),
        ("CD2", 5, (0.535, -0.774, 1.2)),
    ],
    "LYS": [
        ("N", 0, (-0.526, 1.362, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.526, 0, 0)),
        ("O", 3, (0.626, 1.062, 0)),
        ("CB", 0, (-0.524, -0.778, -1.208)),
        ("CG", 4, (0.619, 1.39, 0)),
        ("CD", 5, (0.559, 1.417, 0)),
        ("CE", 6, (0.56, 1.416, 0)),
        ("NZ", 7, (0.554, 1.387, 0)),
    ],
    "MET": [
        ("N", 0, (-0.521, 1.364, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.525, 0, 0)),
        ("O", 3, (0.625, 1.062, 0)),
        ("CB", 0, (-0.523, -0.776, -1.21)),
        ("CG", 4, (0.613, 1.391, 0)),
        ("SD", 5, (0.703, 1.695, 0)),
        ("CE", 6, (0.32, 1.786, 0)),
    ],
    "PHE": [
        ("N", 0, (-0.518, 1.363, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.524, 0, 0)),
        ("O", 3, (0.626, 1.062, 0)),
        ("CB", 0, (-0.525, -0.776, -1.212)),
        ("CG", 4, (0.607, 1.377, 0)),
        ("CD1", 5, (0.709, 1.195, 0)),
        ("CD2", 5, (0.706, -1.196, 0)),
        ("CE1", 5, (2.102, 1.198, 0)),
        ("CE2", 5, (2.098, -1.201, 0)),
        ("CZ", 5, (2.794, -0.003, -0.001)),
    ],
    "PRO": [
        ("N", 0, (-0.566, 1.351, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.527, 0, 0)),
        ("O", 3, (0.621, 1.066, 0)),
        ("CB", 0, (-0.546, -0.611, -1.293)),
        ("CG", 4, (0.382, 1.445, 0)),
        ("CD", 5, (0.477, 1.424, 0)),
    ],
    "SER": [
        ("N", 0, (-0.529, 1.36, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.525, 0, 0)),
        ("O", 3, (0.626, 1.062, 0)),
        ("CB", 0, (-0.518, -0.777, -1.211)),
        ("OG", 4, (0.503, 1.325, 0)),
    ],
    "THR": [
        ("N", 0, (-0.517, 1.364, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.526, 0, 0)),
        ("O", 3, (0.626, 1.062, 0)),
        ("CB", 0, (-0.516, -0.793, -1.215)),
        ("OG1", 4, (0.472, 1.353, 0)),
        ("CG2", 4, (0.55, -0.718, -1.228)),
    ],
    "TRP": [
        ("N", 0, (-0.521, 1.363, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.525, 0, 0)),
        ("O", 3, (0.627, 1.062, 0)),
        ("CB", 0, (-0.523, -0.776, -1.212)),
        ("CG", 4, (0.609, 1.37, 0)),
        ("CD1", 5, (0.824, 1.091, 0)),
        ("CD2", 5, (0.854, -1.148, -0.005)),
        ("NE1", 5, (2.14, 0.69, -0.004)),
        ("CE2", 5, (2.186, -0.678, -0.007)),
        ("CE3", 5, (0.622, -2.53, -0.007)),
        ("CZ2", 5, (3.283, -1.543, -0.011)),
        ("CZ3", 5, (1.715, -3.389, -0.011)),
        ("CH2", 5, (3.028, -2.89, -0.013)),
    ],
    "TYR": [
        ("N", 0, (-0.522, 1.362, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.524, 0, 0)),
        ("O", 3, (0.627, 1.062, 0)),
        ("CB", 0, (-0.522, -0.776, -1.213)),
        ("CG", 4, (0.607, 1.382, 0)),
        ("CD1", 5, (0.716, 1.195, 0)),
        ("CD2", 5, (0.713, -1.194, -0.001)),
        ("CE1", 5, (2.107, 1.2, -0.002)),
        ("CE2", 5, (2.104, -1.201, -0.003)),
        ("CZ", 5, (2.791, -0.001, -0.003)),
        ("OH", 5, (4.168, -0.002, -0.005)),
    ],
    "VAL": [
        ("N", 0, (-0.494, 1.373, 0)),
        ("CA", 0, (0, 0, 0)),
        ("C", 0, (1.527, 0, 0)),
        ("O", 3, (0.627, 1.062, 0)),
        ("CB", 0, (-0.533, -0.795, -1.213)),
        ("CG1", 4, (0.54, 1.429, 0)),
        ("CG2", 4, (0.533, -0.776, 1.203)),
    ],
}


# --------------------------------------------------------------------------
# table construction (numpy, float64 -> float32, exactly as AlphaFold does)
# --------------------------------------------------------------------------
def _rigid_transformation_4x4(ex, ey, translation) -> np.ndarray:
    """AlphaFold's ``_make_rigid_transformation_4x4``: Gram-Schmidt on (ex, ey).

    Note the ``.transpose()``: the basis vectors end up as *columns*, so the
    result maps group-local coordinates to parent coordinates.
    """
    ex = np.asarray(ex, dtype=np.float64)
    ey = np.asarray(ey, dtype=np.float64)
    ex_n = ex / np.linalg.norm(ex)
    ey_n = ey - np.dot(ey, ex_n) * ex_n
    ey_n = ey_n / np.linalg.norm(ey_n)
    ez_n = np.cross(ex_n, ey_n)
    m = np.stack([ex_n, ey_n, ez_n, np.asarray(translation, dtype=np.float64)]).transpose()
    return np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)


def _build_tables():
    n_types = 21  # 20 canonical + UNK
    atom14_to_rigid_group = np.zeros((n_types, 14), dtype=np.int64)
    atom14_mask = np.zeros((n_types, 14), dtype=np.float32)
    atom14_positions64 = np.zeros((n_types, 14, 3), dtype=np.float64)
    default_frame = np.zeros((n_types, 8, 4, 4), dtype=np.float32)
    names: list[list[str]] = [[""] * 14 for _ in range(n_types)]

    for restype, resname in enumerate(RESTYPE_3):
        for idx, (atom_name, group, position) in enumerate(_ATOM14[resname]):
            names[restype][idx] = atom_name
            atom14_to_rigid_group[restype, idx] = group
            atom14_mask[restype, idx] = 1.0
            atom14_positions64[restype, idx] = position

        p = atom14_positions64[restype]
        # number of chi angles == deepest group index beyond psi (group 3)
        num_chi = max(0, int(atom14_to_rigid_group[restype].max()) - 3)

        default_frame[restype, 0] = np.eye(4)   # backbone -> backbone
        default_frame[restype, 1] = np.eye(4)   # pre-omega, a dummy identity
        default_frame[restype, 2] = _rigid_transformation_4x4(   # phi
            ex=p[0] - p[1], ey=[1.0, 0.0, 0.0], translation=p[0])
        default_frame[restype, 3] = _rigid_transformation_4x4(   # psi
            ex=p[2] - p[1], ey=p[1] - p[0], translation=p[2])
        if num_chi >= 1:                                          # chi1
            default_frame[restype, 4] = _rigid_transformation_4x4(
                ex=p[4] - p[1], ey=p[0] - p[1], translation=p[4])
        for k in range(1, num_chi):                               # chi2..chi4
            # every later chi axis starts at its parent group's origin, so the
            # transform is a pure "look along the axis-end atom"
            axis_end = p[4 + k]
            default_frame[restype, 4 + k] = _rigid_transformation_4x4(
                ex=axis_end, ey=[-1.0, 0.0, 0.0], translation=axis_end)

    return (default_frame, atom14_to_rigid_group,
            atom14_positions64.astype(np.float32), atom14_mask, names)


(_DEFAULT_FRAME_NP, _ATOM14_TO_GROUP_NP, _ATOM14_POS_NP, _ATOM14_MASK_NP,
 ATOM14_NAMES) = _build_tables()

restype_rigid_group_default_frame = torch.from_numpy(_DEFAULT_FRAME_NP)  # (21,8,4,4)
restype_atom14_to_rigid_group = torch.from_numpy(_ATOM14_TO_GROUP_NP)    # (21,14)
restype_atom14_rigid_group_positions = torch.from_numpy(_ATOM14_POS_NP)  # (21,14,3)
restype_atom14_mask = torch.from_numpy(_ATOM14_MASK_NP)                  # (21,14)

# The tables are tiny (~60 kB) but gathering them from CPU every call would sync
# on every sampling step, so keep one copy per device.
_DEVICE_CACHE: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}


def _tables(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    key = str(device)
    cached = _DEVICE_CACHE.get(key)
    if cached is None:
        cached = (restype_rigid_group_default_frame.to(device),
                  restype_atom14_to_rigid_group.to(device),
                  restype_atom14_rigid_group_positions.to(device),
                  restype_atom14_mask.to(device))
        _DEVICE_CACHE[key] = cached
    return cached


# --------------------------------------------------------------------------
# angles -> frames -> atoms
# --------------------------------------------------------------------------
def torsion_angles_to_frames(aatype: Tensor, backb_to_global: Rigid,
                             angles: Tensor) -> Rigid:
    """Place all 8 rigid groups of each residue in global coordinates.

    Args:
        aatype: ``(L,)`` int64 in 0..20.
        backb_to_global: `Rigid` of shape ``(L,)``.
        angles: ``(L, 7, 2)`` unit ``(sin, cos)`` pairs for
            omega, phi, psi, chi1..chi4.
    Returns:
        `Rigid` of shape ``(L, 8)``.
    """
    default_frame, _, _, _ = _tables(aatype.device)
    m = default_frame[aatype].to(angles.dtype)            # (L, 8, 4, 4)
    default = Rigid(m[..., :3, :3], m[..., :3, 3])

    # group 0 is the backbone itself, which never rotates -> (sin, cos) = (0, 1)
    lead = angles.shape[:-2]
    sin = torch.cat([angles.new_zeros(lead + (1,)), angles[..., 0]], dim=-1)
    cos = torch.cat([angles.new_ones(lead + (1,)), angles[..., 1]], dim=-1)
    zero = torch.zeros_like(sin)
    one = torch.ones_like(sin)
    rot_x = torch.stack([                                  # (L, 8, 3, 3)
        torch.stack([one, zero, zero], dim=-1),
        torch.stack([zero, cos, -sin], dim=-1),
        torch.stack([zero, sin, cos], dim=-1),
    ], dim=-2)
    # the torsion only spins the group about its own x-axis; the origin is fixed
    all_frames = Rigid(default.rotation @ rot_x, default.translation)

    # chi2..chi4 are defined relative to the *previous* chi, so chain them
    chi1 = all_frames[:, 4]
    chi2 = chi1 @ all_frames[:, 5]
    chi3 = chi2 @ all_frames[:, 6]
    chi4 = chi3 @ all_frames[:, 7]
    to_backb = Rigid(
        torch.cat([all_frames.rotation[:, :5], chi2.rotation[:, None],
                   chi3.rotation[:, None], chi4.rotation[:, None]], dim=1),
        torch.cat([all_frames.translation[:, :5], chi2.translation[:, None],
                   chi3.translation[:, None], chi4.translation[:, None]], dim=1))
    return backb_to_global[:, None] @ to_backb


def frames_and_literature_positions_to_atom14_pos(aatype: Tensor,
                                                  all_frames: Rigid) -> Tensor:
    """Scatter each atom's literal position through its own group's frame.

    Args:
        aatype: ``(L,)`` int64 in 0..20.
        all_frames: `Rigid` of shape ``(L, 8)`` from `torsion_angles_to_frames`.
    Returns:
        ``(L, 14, 3)``; atoms the residue does not have are exactly zero.
    """
    _, atom14_to_group, atom14_pos, atom14_mask = _tables(aatype.device)
    dtype = all_frames.rotation.dtype
    group_idx = atom14_to_group[aatype]                              # (L, 14)
    # one-hot pick of the owning group, then a contraction - cheaper to write
    # than a gather and identical to the reference
    group_mask = (group_idx[..., None] == torch.arange(8, device=aatype.device)
                  ).to(dtype)                                        # (L, 14, 8)
    rot = torch.einsum("lgij,lag->laij", all_frames.rotation, group_mask)
    trans = torch.einsum("lgi,lag->lai", all_frames.translation, group_mask)
    literal = atom14_pos[aatype].to(dtype)                           # (L, 14, 3)
    pred = torch.einsum("laij,laj->lai", rot, literal) + trans
    return pred * atom14_mask[aatype].to(dtype)[..., None]


def single_protein_sidechains(aatype: Tensor, frames: Rigid,
                              angles: Tensor) -> tuple[Tensor, Rigid]:
    """``(atom14 positions (L, 14, 3), all group frames (L, 8))``."""
    all_frames = torsion_angles_to_frames(aatype, frames, angles)
    return frames_and_literature_positions_to_atom14_pos(aatype, all_frames), all_frames
