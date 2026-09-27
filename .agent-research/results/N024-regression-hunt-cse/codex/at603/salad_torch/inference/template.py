"""``--template <pdb_path>,<constring>`` motif conditioning.

`notes/specs/06_generation_flags.md` §2. Two genuinely separate pieces:

  * `parse_constring` - which residues are "F"ixed (came from the template
    structure) and which chain each belongs to. Pure string parsing.
  * `read_pdb` - an actual PDB reader. The port had none (`io/pdb.py` only
    writes); this is new code, a thin Biopython-based reproduction of
    upstream's `from_pdb_string` / `_from_bio_structure` recipe.

`parse_template` combines the two into the `(aatype, dmap, omap, dmap_mask,
chain_index)` tuple that `salad_torch.modules.condition.prepare_condition`
already knows how to consume unchanged - no changes needed there, and none
made here.

Not ported: the trunk-side embedding of `omap` (`c.use_omap`) is dead for
every config this port supports (see the package docstring) - `omap` is still
built here (cheap, and `dmap`/`dmap_mask`/`aa_condition` share the same
parsing pass) but nothing consumes it beyond the harmless zero-init default.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..geometry.frames import extract_aa_frames, positions_to_ncacocb
from ..io import residue_constants as rc
from ..io.atom_layout import atom37_to_atom14
from ..io.pdb import Protein

__all__ = ["parse_constring", "read_pdb", "parse_template"]


def parse_constring(constring: str) -> tuple[Tensor, Tensor, Tensor]:
    """``"FFF__FFF:FF"`` -> ``(conmask, chain_index, dmap_mask)``. §2.1.

    ``:`` separates chains; ``"F"`` marks a residue as template-conditioned
    ("Fixed"); any other character (conventionally ``"_"``) marks it free.
    `dmap_mask[i, j]` is true only where *both* `i` and `j` are `"F"` -
    the distance map is only trustworthy between two motif residues that both
    actually came from the template.
    """
    conmask: list[bool] = []
    chain_index: list[int] = []
    chain = 0
    for c in constring:
        if c == "F":
            conmask.append(True)
            chain_index.append(chain)
        elif c == ":":
            chain += 1
        else:
            conmask.append(False)
            chain_index.append(chain)
    conmask_t = torch.tensor(conmask, dtype=torch.bool)
    chain_t = torch.tensor(chain_index, dtype=torch.long)
    dmap_mask = conmask_t[:, None] & conmask_t[None, :]
    return conmask_t, chain_t, dmap_mask


def read_pdb(path: str) -> Protein:
    """Parse a PDB file into a `Protein`. New code - the port had no reader.

    Reproduces upstream's `from_pdb_string` recipe (Biopython `PDBParser` ->
    per-residue atom37 extraction), not "improved": multi-model files and
    insertion codes are refused (`ValueError`), non-standard residues become
    `UNK`/`X` (restype 20), unrecognised atom names are silently ignored, and
    a residue with zero recognised atoms is dropped. Chain ids are remapped to
    0-based integers **alphabetically** (`sorted(unique(...))`, matching
    `np.unique`), not by file order.

    Heteroatom records (e.g. waters) are not filtered out here - upstream's
    own recipe doesn't either; a `"HOH"` residue just becomes `UNK` and
    contributes whatever of its atoms happen to share a name with
    `residue_constants.atom_types` (typically just `O`). Keep template PDBs to
    ATOM records only if that matters for your use case.
    """
    from Bio.PDB import PDBParser  # imported lazily: only --template needs it

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", path)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"{path}: multi-model PDBs are not supported for --template "
            f"({len(models)} models found)")

    aatypes: list[int] = []
    positions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    bfactors: list[np.ndarray] = []
    resseqs: list[int] = []
    chain_letters: list[str] = []

    for chain in models[0]:
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"{path}: residue {res.id} has an insertion code, which "
                    "is not supported for --template")
            resname = res.resname.strip()
            one_letter = rc.restype_3to1.get(resname, "X")
            restype_idx = rc.restype_order.get(one_letter, rc.restype_num)

            pos = np.zeros((rc.atom_type_num, 3), dtype=np.float32)
            mask = np.zeros((rc.atom_type_num,), dtype=np.float32)
            bfac = np.zeros((rc.atom_type_num,), dtype=np.float32)
            for atom in res:
                name = atom.get_name()
                if name not in rc.atom_order:
                    continue
                idx = rc.atom_order[name]
                pos[idx] = atom.coord
                mask[idx] = 1.0
                bfac[idx] = atom.get_bfactor()
            if mask.sum() < 0.5:
                continue

            aatypes.append(restype_idx)
            positions.append(pos)
            masks.append(mask)
            bfactors.append(bfac)
            resseqs.append(res.id[1])
            chain_letters.append(chain.id)

    if not aatypes:
        raise ValueError(f"{path}: no residues with recognised atoms found")

    unique_letters = sorted(set(chain_letters))
    letter_to_index = {cid: n for n, cid in enumerate(unique_letters)}
    chain_index = np.array([letter_to_index[c] for c in chain_letters], dtype=np.int64)

    return Protein(
        atom_positions=np.stack(positions, axis=0),
        aatype=np.array(aatypes, dtype=np.int64),
        atom_mask=np.stack(masks, axis=0),
        residue_index=np.array(resseqs, dtype=np.int64),
        chain_index=chain_index,
        b_factors=np.stack(bfactors, axis=0),
    )


def parse_template(spec: str) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``"<pdb_path>,<constring>"`` -> ``(aatype, dmap, omap, dmap_mask, chain_index)``.

    §2.2-2.3. `chain_index` (returned) comes from the **constring**, not the
    PDB file - the template PDB's own chain boundaries are irrelevant once its
    coordinates are extracted sequentially in file-residue order and
    re-indexed against `conmask`.

    The number of `"F"` characters in `constring` must exactly equal the
    number of residues `read_pdb` recovers from the PDB (a template PDB with
    more or fewer usable residues than fixed positions in the constring raises
    a shape-mismatch, not a graceful error - upstream never validated this
    either, see §0/§2 of the spec).
    """
    path, constring = spec.split(",", 1)
    conmask, chain_index, dmap_mask = parse_constring(constring)
    size = conmask.shape[0]

    protein = read_pdb(path)
    aatype_parsed = torch.from_numpy(protein.aatype)
    atom_positions = torch.from_numpy(protein.atom_positions)
    atom_mask = torch.from_numpy(protein.atom_mask)

    atom14, _ = atom37_to_atom14(aatype_parsed, atom_positions, atom_mask)
    ncacocb = positions_to_ncacocb(atom14)   # (n_provided, 5, 3)
    cb_provided = ncacocb[:, 4]

    n_provided = int(conmask.sum())
    if cb_provided.shape[0] != n_provided:
        raise ValueError(
            f"--template: constring has {n_provided} 'F' positions but the "
            f"PDB provided {cb_provided.shape[0]} residues - these must match")

    cb = torch.zeros((size, 3), dtype=cb_provided.dtype)
    cb[conmask] = cb_provided
    ncacocb_full = torch.zeros((size, 5, 3), dtype=ncacocb.dtype)
    ncacocb_full[conmask] = ncacocb
    aatype = torch.full((size,), 20, dtype=torch.long)
    aatype[conmask] = aatype_parsed

    frames, _ = extract_aa_frames(ncacocb_full)
    rot = frames.rotation                            # (size, 3, 3)
    inv_rot = rot.transpose(-1, -2)
    omap = inv_rot[:, None] @ rot[None, :]            # (size, size, 3, 3)
    omap = omap.reshape(size, size, 9)

    dmap = torch.linalg.norm(cb[:, None] - cb[None, :], dim=-1)

    return aatype, dmap, omap, dmap_mask, chain_index
