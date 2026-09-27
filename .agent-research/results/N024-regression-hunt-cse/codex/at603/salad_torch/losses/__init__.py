"""salad `default_vp` training losses - public surface.

See `notes/specs/03_training_losses.md` for the full specification and
`losses.total.total_loss` for the master entry point used by a training loop.
Individual terms remain importable from their owning submodule
(`losses.structure`, `losses.sequence`, `losses.violation`) for unit testing.
"""
from __future__ import annotations

from .sequence import aa_loss, distogram_loss, dssp_loss
from .structure import (backbone_cb_distance, base_weight, do_clip, fape_loss,
                        fape_neighbours, fape_trajectory_loss, late_mask, local_loss,
                        pos_loss, pos_trajectory_loss, rotation_loss,
                        rotation_trajectory_loss, sparse_fape, x_loss)
from .total import total_loss
from .violation import (between_residue_bond_loss, custom_clash_loss,
                        custom_structural_violation_loss, find_structural_violations,
                        get_atom14_mask, load_stereo_chemical_props,
                        make_atom14_dists_bounds, violation_loss,
                        within_residue_violations)

__all__ = [
    # sequence
    "aa_loss", "dssp_loss", "distogram_loss",
    # structure
    "base_weight", "late_mask", "do_clip",
    "pos_loss", "pos_trajectory_loss", "x_loss",
    "rotation_loss", "rotation_trajectory_loss",
    "backbone_cb_distance", "fape_neighbours", "sparse_fape",
    "fape_loss", "fape_trajectory_loss", "local_loss",
    # violation
    "load_stereo_chemical_props", "make_atom14_dists_bounds", "get_atom14_mask",
    "between_residue_bond_loss", "custom_clash_loss", "within_residue_violations",
    "custom_structural_violation_loss", "find_structural_violations", "violation_loss",
    # total
    "total_loss",
]
