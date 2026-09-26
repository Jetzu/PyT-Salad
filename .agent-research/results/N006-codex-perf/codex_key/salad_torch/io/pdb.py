"""PDB output.

Port of `salad.aflib.common.protein.Protein` / `to_pdb` plus the eval script's
writer (`eval_noise_schedule_benchmark.py:656-671`). Pure string formatting over
numpy arrays - torch tensors are converted at the boundary.

Three things about salad's PDB files surprise people, and all three are
reproduced on purpose:

* **Atoms are written in atom37 order**, so CB comes out *before* O. Unusual,
  but it round-trips and every parser accepts it.
* **Residue numbering starts at 0**, with a +50 gap between chains that is
  measured from the previous chain's last residue index, not its length. See
  `parse_num_aa`.
* **The b-factor column holds the model's max amino-acid probability**, in
  [0, 1], not a temperature factor.

The writer emits a single MODEL/ENDMDL pair, always closes with a TER, and pads
every line to 80 columns.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from torch import Tensor

from . import residue_constants as rc
from .atom_layout import atom14_to_atom37, get_atom37_mask

__all__ = ["PDB_CHAIN_IDS", "PDB_MAX_CHAINS", "Protein", "to_pdb",
           "max_aa_probability", "snapshot_to_protein", "write_pdb"]

#: `protein.py:36`. 62 single-character chain ids; more than that cannot be written.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62


def _np(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclasses.dataclass(frozen=True)
class Protein:
    """`protein.py:65-98`. Field order is positional-construction order."""

    atom_positions: np.ndarray   # (N, 37, 3) float32
    aatype: np.ndarray           # (N,)  int, 0..20 (20 = 'X'/UNK)
    atom_mask: np.ndarray        # (N, 37) float
    residue_index: np.ndarray    # (N,)  int
    chain_index: np.ndarray      # (N,)  int
    b_factors: np.ndarray        # (N, 37) float

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} "
                "chains because these cannot be written to PDB format.")


def _chain_end(atom_index: int, end_resname: str, chain_name: str,
               residue_index: int) -> str:
    """`protein.py:231-235`. The six spaces after the serial number are literal."""
    return (f'{"TER":<6}{atom_index:>5}      {end_resname:>3} '
            f'{chain_name:>1}{residue_index:>4}')


def to_pdb(prot: Protein) -> str:
    """Render a `Protein` as a PDB string. `protein.py:238-315`."""
    restypes = rc.restypes + ["X"]

    def res_1to3(r: int) -> str:
        return rc.restype_1to3.get(restypes[r], "UNK")

    atom_types = rc.atom_types
    pdb_lines: list[str] = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > rc.restype_num):
        raise ValueError("Invalid aatypes.")

    # chain ids are assigned from the *sorted unique* chain indices, so a gap in
    # chain_index skips a letter rather than compacting
    chain_ids = {}
    for i in np.unique(chain_index):
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {PDB_MAX_CHAINS} chains.")
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    for i in range(aatype.shape[0]):
        # close the *previous* chain before opening this one
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(
                atom_index, res_1to3(aatype[i - 1]),
                chain_ids[chain_index[i - 1]], residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1  # the TER record consumes a serial number

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
                atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            # first character of the atom name; correct for C/N/O/S, which is
            # all the heavy-atom table contains
            element = atom_name[0]
            charge = ""
            atom_line = (f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                         f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                         f"{residue_index[i]:>4}{insertion_code:>1}   "
                         f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                         f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                         f"{element:>2}{charge:>2}")
            pdb_lines.append(atom_line)
            atom_index += 1

    pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                                chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"


def max_aa_probability(aa_logits: Tensor) -> Tensor:
    """``softmax(aa).max(-1)`` -> ``(N,)``, the b-factor column.

    `aa_logits` is already ``log_softmax`` output
    (``noise_schedule_benchmark.py:1564``), so this softmax is mathematically an
    identity and the result is simply the max predicted probability. Upstream
    applies it anyway (``eval_...:622``); doing the same keeps the last ulp
    matching, which ``aa_logits.exp().max(-1)`` would not.
    """
    return torch.softmax(aa_logits, dim=-1).max(dim=-1).values


def snapshot_to_protein(snapshot: dict, *, chain_index: Tensor | None = None,
                        apply_residue_mask: bool = False) -> Protein:
    """Turn one sampler snapshot into a `Protein`. ``eval_...:659-668``.

    `chain_index` should be the **pre-``--merge_chains``** assignment: a merged
    run still writes separate PDB chains with TER records between them, even
    though the model saw one chain. Defaults to the snapshot's own
    `chain_index`, which is the post-merge one.

    `apply_residue_mask=False` reproduces the eval script, which uses
    ``get_atom37_mask(aatype)`` alone. Setting it True multiplies in
    ``data["mask"]``, the `inference/data.py:151` variant, which drops padding
    residues.
    """
    aatype = snapshot["aatype"]
    atom37 = atom14_to_atom37(snapshot["atom_pos"], aatype)
    atom37_mask = get_atom37_mask(aatype)
    if apply_residue_mask:
        atom37_mask = snapshot["mask"].to(atom37_mask.dtype)[:, None] * atom37_mask

    maxprob = max_aa_probability(snapshot["aa_logits"])
    # (N, 1) * ones(N, 37): the same scalar in all 37 slots, unmasked. to_pdb
    # only reads slots the atom mask keeps, so the extras never surface.
    b_factors = maxprob[:, None] * torch.ones_like(atom37_mask)

    if chain_index is None:
        chain_index = snapshot["chain_index"]

    return Protein(
        _np(atom37).astype(np.float32),
        _np(aatype),
        _np(atom37_mask),
        _np(snapshot["residue_index"]),
        _np(chain_index),
        _np(b_factors),
    )


def write_pdb(path: str, snapshot: dict, *, chain_index: Tensor | None = None,
              apply_residue_mask: bool = False) -> str:
    """Write one snapshot to `path` and return the PDB string."""
    pdb_string = to_pdb(snapshot_to_protein(
        snapshot, chain_index=chain_index, apply_residue_mask=apply_residue_mask))
    with open(path, "wt") as handle:
        handle.write(pdb_string)
    return pdb_string
