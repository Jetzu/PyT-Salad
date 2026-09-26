"""Amino-acid and atom-naming constants, ported from `salad.aflib.common.residue_constants`.

Only the five *literal* tables are copied here (verbatim, so a diff against
upstream is trivial); everything else is rebuilt at import time with the same
builders `salad.aflib.model.all_atom_multimer` uses. Retyping the derived
`(21, 37)` index maps by hand would be an excellent way to introduce a silent
one-atom-off bug, so we don't.

Two orderings coexist and they are *not* the same:

* **atom37** (`atom_types`) is `N, CA, C, CB, O, CG, ...` - CB at index 3, O at 4.
* **atom14** (`restype_name_to_atom14_names`) is `N, CA, C, O, CB, ...`.

Hence the `[0, 1, 2, 4, 3, ...]` swap at the head of both conversion maps. The
PDB writer walks atom37 order, which is why salad's output lists CB *before* O.

Absent atoms use index 0 as a sentinel in both maps, so a gather alone
duplicates atom N into every empty slot; always apply the companion mask.
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "atom_types", "atom_order", "atom_type_num",
    "restypes", "restypes_with_x", "restype_order", "restype_num",
    "unk_restype_index", "restype_1to3", "restype_3to1",
    "restype_name_to_atom14_names", "residue_atoms",
    "RESTYPE_ATOM37_MASK", "RESTYPE_ATOM14_MASK",
    "RESTYPE_ATOM37_TO_ATOM14", "RESTYPE_ATOM14_TO_ATOM37",
    "table_to",
]

# ---------------------------------------------------------------------------
# Literal tables - verbatim from residue_constants.py
#   residue_atoms                :338-361
#   atom_types                   :492-497
#   restype_name_to_atom14_names :505-528
#   restypes                     :535-538
#   restype_1to3                 :590-611
# ---------------------------------------------------------------------------
residue_atoms = {
    'ALA': ['C', 'CA', 'CB', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3',
            'CH2', 'N', 'NE1', 'O'],
    'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O',
            'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O']
}

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],

}

restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

# ---------------------------------------------------------------------------
# Derived scalar tables - residue_constants.py:498-499, 539-541, 647
# ---------------------------------------------------------------------------
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37

restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20
unk_restype_index = restype_num  # catch-all index for unknown restypes
restypes_with_x = restypes + ["X"]
restype_3to1 = {v: k for k, v in restype_1to3.items()}


# ---------------------------------------------------------------------------
# Derived index/mask tables - all_atom_multimer.py:85-142
# Row 20 is UNK and is deliberately all-zero / all-index-0.
# ---------------------------------------------------------------------------
def _make_restype_atom37_mask() -> Tensor:
    mask = torch.zeros((21, 37), dtype=torch.float32)
    for restype, letter in enumerate(restypes):
        for atom_name in residue_atoms[restype_1to3[letter]]:
            mask[restype, atom_order[atom_name]] = 1.0
    return mask


def _make_restype_atom14_mask() -> Tensor:
    rows = [[1.0 if name else 0.0
             for name in restype_name_to_atom14_names[restype_1to3[rt]]]
            for rt in restypes]
    rows.append([0.0] * 14)
    return torch.tensor(rows, dtype=torch.float32)


def _make_restype_atom37_to_atom14() -> Tensor:
    rows = []
    for rt in restypes:
        names = restype_name_to_atom14_names[restype_1to3[rt]]
        # built from the padded 14-name list, so '' maps to the last empty slot;
        # unreachable, because '' is never a member of atom_types. Upstream quirk.
        name_to_idx14 = {name: i for i, name in enumerate(names)}
        rows.append([name_to_idx14.get(name, 0) for name in atom_types])
    rows.append([0] * 37)
    return torch.tensor(rows, dtype=torch.int32)


def _make_restype_atom14_to_atom37() -> Tensor:
    rows = [[atom_order[name] if name else 0
             for name in restype_name_to_atom14_names[restype_1to3[rt]]]
            for rt in restypes]
    rows.append([0] * 14)
    return torch.tensor(rows, dtype=torch.int32)


RESTYPE_ATOM37_MASK = _make_restype_atom37_mask()          # (21, 37) f32
RESTYPE_ATOM14_MASK = _make_restype_atom14_mask()          # (21, 14) f32
RESTYPE_ATOM37_TO_ATOM14 = _make_restype_atom37_to_atom14()  # (21, 37) i32
RESTYPE_ATOM14_TO_ATOM37 = _make_restype_atom14_to_atom37()  # (21, 14) i32


# ---------------------------------------------------------------------------
# Device cache
# ---------------------------------------------------------------------------
_DEVICE_CACHE: dict[tuple[int, torch.device], Tensor] = {}


def table_to(table: Tensor, device: torch.device | str | None) -> Tensor:
    """Return `table` on `device`, memoised.

    The four lookup tables are tiny but are indexed once per residue per PDB
    write and once per model call; re-uploading them to the GPU every time is
    pure latency.
    """
    if device is None:
        return table
    device = torch.device(device)
    if table.device == device:
        return table
    key = (id(table), device)
    cached = _DEVICE_CACHE.get(key)
    if cached is None:
        cached = table.to(device)
        _DEVICE_CACHE[key] = cached
    return cached
