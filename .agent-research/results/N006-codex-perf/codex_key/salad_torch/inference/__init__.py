"""CLI-level inference helpers for `scripts/generate.py`.

Everything in this package is pure CLI/setup-level composition of already
exported model pieces (`apply_diffusion`, `model.predict`, the `sample()`
hooks) - see `notes/specs/06_generation_flags.md`. Nothing here needs, or is
allowed, to modify `salad_torch/modules/diffusion.py`, `modules/fast.py`,
`geometry/segment.py` or `modules/blocks.py`.

Deliberately NOT ported (see `notes/specs/06_generation_flags.md` for why):

* ``config.symm`` / ``update_positions(symm=...)`` - dead code in every
  released config; `--sym`'s post-hoc symmetrisers below are the only two
  live symmetry call sites.
* ``salad/inference/symmetry.py``'s class-based ``Screw`` - unused by the eval
  script this CLI mirrors.
* the trunk-side ``use_omap`` embedding (`DiffusionPairFeatures.linear_omap`)
  - unreachable for every config the port supports (`default_vp`,
  `default_vp_scaled`, `default_ve_scaled`, `multimotif_vp` all leave
  ``use_omap`` unset).
"""
from __future__ import annotations

from .options import (DSSP_CODE, decode_sequence, parse_dssp, parse_dssp_mean,
                      random_dssp, random_dssp_mean)
from .postprocess import reorder_amino_acids
from .symmetry import SymmetrySpec, cyclic_centering, parse_sym, rot_y, screw_replicate
from .template import parse_constring, parse_template, read_pdb

__all__ = [
    "SymmetrySpec", "parse_sym", "rot_y", "cyclic_centering", "screw_replicate",
    "parse_constring", "read_pdb", "parse_template",
    "reorder_amino_acids",
    "DSSP_CODE", "parse_dssp", "parse_dssp_mean", "random_dssp",
    "random_dssp_mean", "decode_sequence",
]
