"""``--order_agnostic``'s output-side fixup: re-thread residues by geometry.

`notes/specs/06_generation_flags.md` §4.2. The input-side half (`resi = -1`,
`chain = 0`, so the network never sees residue order) is CLI setup, done in
`scripts/generate.py`; this module is the post-hoc, PDB-writing-only half that
walks the C->N backbone graph to recover a sensible residue order and renumber
accordingly. It never feeds back into the sampling loop.
"""
from __future__ import annotations

import numpy as np

from ..io import residue_constants as rc
from ..io.pdb import Protein

__all__ = ["reorder_amino_acids"]


def reorder_amino_acids(protein: Protein, *, max_iterations_factor: int = 2) -> Protein:
    """Re-thread `protein`'s residues by following C->N backbone contacts.

    A genuine graph walk (not a sort): follow each residue's outgoing C->N
    bond to the next; whenever the current walk closes a cycle or hits a
    dead end (a residue with no outgoing C->N bond within 2.5 A), restart at a
    fresh, non-terminal, unvisited residue. `new_chain_index` is
    unconditionally all-zero - this assumes/produces a single output chain
    regardless of how many chains went in (matches upstream; a natural
    consequence of `--order_agnostic` already having merged everything into
    one chain on the input side).

    Safety net upstream lacks: a residue that is both a terminus (`stop`) and
    never any other residue's `next_of` target can never be reached or
    selected as a fresh start, so `visited.all()` never becomes true and the
    naive algorithm hangs forever re-walking already-visited segments. This
    is plausible for a raw diffusion sample (a local geometry defect breaking
    backbone connectivity), not a theoretical concern - so a hard iteration
    cap (`max_iterations_factor * L`) raises a clear `RuntimeError` instead of
    hanging.
    """
    atom_positions = protein.atom_positions
    L = atom_positions.shape[0]
    n_idx, c_idx = rc.atom_order["N"], rc.atom_order["C"]

    N = atom_positions[:, n_idx]
    C = atom_positions[:, c_idx]
    c_to_n = np.linalg.norm(C[:, None, :] - N[None, :, :], axis=-1)
    np.fill_diagonal(c_to_n, np.inf)
    adjacency = c_to_n < 2.5

    visited = np.zeros(L, dtype=bool)
    weight = np.ones(L, dtype=np.int64)
    next_of = np.argmax(adjacency, axis=1)
    stop = ~adjacency.any(axis=1)

    count = 0
    index = 0
    vinloop: list[int] = []
    max_iterations = max_iterations_factor * L

    for _ in range(max_iterations):
        if index in vinloop:
            if visited.all():
                break
            index = int(np.argmax((~visited) & (~stop)))
            vinloop = []
        visited[index] = True
        vinloop.append(index)
        weight[index] = count
        count += 1
        if stop[index]:
            if visited.all():
                break
            index = int(np.argmax((~visited) & (~stop)))
            count += 1
        else:
            index = int(next_of[index])
    else:
        raise RuntimeError(
            f"reorder_amino_acids: exceeded {max_iterations} iterations without "
            f"visiting all {L} residues ({int((~visited).sum())} unvisited) - "
            "the backbone graph likely has an isolated residue (no incoming or "
            "outgoing C-N bond within 2.5 A) that can never be reached or "
            "selected as a fresh walk start.")

    order = np.argsort(weight, kind="stable")
    new_residue_index = weight[order]
    new_chain_index = np.zeros(L, dtype=protein.chain_index.dtype)

    return Protein(
        atom_positions[order],
        protein.aatype[order],
        protein.atom_mask[order],
        new_residue_index,
        new_chain_index,
        protein.b_factors[order],
    )
