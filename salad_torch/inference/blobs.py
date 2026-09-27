"""``--blobs``/``--blob_scale`` shape conditioning. `notes/specs/08_shape_conditioning.md`.

Ported from upstream's `salad/training/eval_ve_shape.py::parse_blob`. There is
no useful `salad/inference/shape.py` counterpart to build on - that module is
two bare `import jax`/`import jax.numpy as jnp` lines and nothing else, dead.
VE-model only: `eval_ve_shape.py` is upstream's whole generation loop for
`default_ve_scaled` (`preconditioning == "edm_scaled"`), and `parse_blob`
builds the model's *initial* point cloud (``data["pos"]`` before the first
noising step) directly from a CSV of blob centres, in place of `--num_aa`'s
all-zero cloud.

Two pieces here, matching upstream's two call sites:

* `parse_blobs` - the CSV -> point-cloud + flat `resi`/`chain` construction
  (`eval_ve_shape.py::parse_blob`, called once, before the design loop).
* `replace_to_blob_anchor` - the per-step "hard replace" correction
  (`eval_ve_shape.py`'s `__main__`, inline, no name upstream), applied after
  every step (by default; gated by `--replace_threshold`) through
  `sample()`'s existing `post_step_fn` hook - the same seam `--sym`'s Hook C
  (`salad_torch/inference/symmetry.py`) already uses.

Why the point cloud alone is not the whole feature: at ``raw_t=1`` (the first
step), `apply_diffusion`'s edm branch computes
``pos_noised = data["pos"] + t * centred_normal`` (`diffusion/forward.py`,
`diffuse_coordinates_edm`) - i.e. each blob's repeated coordinate plus a wide,
~isotropic Gaussian offset. That single line *is* "VE noise arranges residues
in a normal distribution around each location"; nothing else needs to change
for it. `replace_to_blob_anchor` is the second, independent mechanism that
keeps the trajectory from drifting away from the blob layout over the
remaining ~499 steps.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = ["parse_blobs", "replace_to_blob_anchor"]


def parse_blobs(path: str, *, augment_size: int, blob_scale: float = 1.0
                ) -> tuple[int, Tensor, Tensor, Tensor, bool, Tensor]:
    """``eval_ve_shape.py::parse_blob``. CSV rows are ``x,y,z,num_aa``.

    Each row places ``num_aa`` consecutive residues at ``(x, y, z)``: every
    atom of every one of those residues starts at the exact same literal
    point (see the module docstring for how that spreads out at the first
    noising step).

    Faithfully reproduces two upstream quirks. Neither is fixed here, per
    this port's convention of reproducing upstream bugs rather than
    silently improving on them:

    * ``next(f)`` unconditionally discards the first line - there is no
      check that it actually reads ``"x,y,z,num_aa"``. A headerless CSV
      silently loses its first blob to this.
    * ``*xyz, num_aa = line.strip().split(",")`` - a row with other than 4
      comma-separated fields is not rejected; it silently changes how many
      elements land in `xyz` (a non-3-vector coordinate raises later, inside
      `np.array`/the broadcast against `init_pos`, rather than at parse time).

    Unlike `--num_aa`/`parse_num_aa`, there is no per-segment 50-residue gap
    and no cyclic support: `resi = arange(num_aa)` is one flat, contiguous
    numbering and `chain = zeros(num_aa)` is one unnumbered chain, regardless
    of how many blob rows contributed to it - upstream's `parse_blob` builds
    these itself rather than calling `parse_num_aa`, and this mirrors that
    construction exactly, not the general `--num_aa` grammar.

    `blob_scale` multiplies every coordinate (`--blob_scale`, upstream
    default 1.8). Upstream applies it to `init_pos` from the caller, after
    `parse_blob` returns; folded in here instead - a scalar multiply is
    applied to the same tensor either way, and doing it before deriving
    `num_aa`/`resi`/`chain` from `init_pos.shape` changes nothing, since a
    multiply never changes shape.

    Returns ``(num_aa, init_pos, resi, chain, is_cyclic, cyclic_mask)`` - the
    same 6-tuple shape as `salad_torch.diffusion.schedules.parse_num_aa`,
    matching upstream's own `parse_blob` return signature 1:1 (upstream:
    ``num_aa, init_pos, resi, chain, is_cyclic, cyclic_mask``). `init_pos`
    has shape ``(num_aa, 5 + augment_size, 3)``, float32, on the CPU - moved
    to the target device by the caller, matching how `parse_num_aa`'s
    `resi`/`chain` are moved inside `make_initial_data` rather than here.
    """
    atom_count = 5 + int(augment_size)
    chunks = []
    with open(path) as fh:
        next(fh)   # unconditionally skips line 1 - see docstring
        for line in fh:
            *xyz, count_s = line.strip().split(",")
            count = int(count_s)
            coord = np.array([float(v) for v in xyz], dtype=np.float32)
            block = np.repeat(coord[None, None], count, axis=0)
            block = np.repeat(block, atom_count, axis=1)
            chunks.append(block)
    init_pos = np.concatenate(chunks, axis=0) * np.float32(blob_scale)
    num_aa = init_pos.shape[0]

    init_pos_t = torch.from_numpy(init_pos)
    resi = torch.arange(num_aa, dtype=torch.int32)
    chain = torch.zeros(num_aa, dtype=torch.int32)
    cyclic_mask = torch.zeros(num_aa, dtype=torch.bool)
    return num_aa, init_pos_t, resi, chain, False, cyclic_mask


def replace_to_blob_anchor(pos: Tensor, init_pos: Tensor, *, chunk: int = 100) -> Tensor:
    """The per-step hard replace, upstream ``eval_ve_shape.py``'s ``__main__``
    (inline, no name upstream): re-centre each `chunk`-residue block's
    predicted CA centroid onto that block's mean blob anchor, then translate
    every atom of every residue in the block by the same vector - so the
    network's own *local* geometry (each atom's offset from its block's own
    centroid) is preserved, only the block's placement in world space moves.

    ``chunk=100`` is hardcoded upstream, not derived from the blob CSV's own
    `num_aa` column - it is a bare assumption that every blob spans exactly
    100 residues (true of all four of upstream's shipped `letters/*.csv`
    example files). Reproduced as a hardcoded constant, not generalised to
    "one chunk per blob": if the caller's total residue count is not a
    multiple of 100, the `reshape` below raises `RuntimeError`, exactly as
    upstream's ``pos[:, 1].reshape(-1, 100, 3)`` would - not caught or
    validated, per "reproduce the crash, don't add validation upstream never
    had."

    Gating (``raw_t >= --replace_threshold``, default 0.0 = every step,
    matching upstream's own default) is the caller's job
    (`scripts/generate.py`'s `post_step_fn`), matching how upstream's own
    ``if raw_t >= opt.replace_threshold:`` sits directly in `__main__`, not in
    a helper function.
    """
    n = pos.shape[0]
    centers = pos[:, 1].reshape(-1, chunk, 3).mean(dim=1, keepdim=True)
    centers = centers.repeat(1, chunk, 1).reshape(n, 3)
    return pos - centers[:, None, :] + init_pos
