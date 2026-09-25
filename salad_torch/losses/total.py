"""The master `default_vp` training loss.

Assembles every term in `losses.structure`/`losses.sequence`/`losses.violation`
with the exact config-driven weights and accumulation order of
`notes/specs/03_training_losses.md` §6, for `diffusion_kind="vpfixed"`,
`preconditioning="vp"`, `diffusion_time_scale="cosine"` (`default_vp`).

Two headline facts about this exact config (spec §0), both still implemented
generally and only true as a CONSEQUENCE of `default_vp`'s settings:
  (a) `c.kabsch` is unset -> every position/atom/FAPE term compares raw,
      un-aligned world-frame coordinates (`losses.geometry.align` is never
      called on this path - `total_loss` raises if a caller's config sets
      `kabsch` truthy, rather than silently ignoring it).
  (b) `preconditioning_scale_factors(c, t_pos, sigma_data)["loss"]` is the
      literal Python int `1` for `preconditioning=="vp"` - so
      `diffusion_weight == base_weight` exactly; there is no time-dependent
      reweighting of `pos`/`pos_trajectory` at all.

Only wires the terms ACTIVE for `default_vp` (spec §8 lists what's
deliberately left out: `aa_trajectory`, `distogram_trajectory`, `soft_lddt`/
`lddt`, `x_late`, `mask_atom14`, the EDM/flow noising branches). Every one of
those needs its own config flag and its own wiring before it belongs here -
consult the spec before adding one.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..diffusion.precond import preconditioning_scale_factors
from ..geometry.frames import extract_aa_frames
from . import sequence as SEQ
from . import structure as STRUCT
from . import violation as VIOL

__all__ = ["total_loss"]

# hardcoded literals with no config field (spec §7.3, §7.9, §7.10)
_DSSP_WEIGHT = 1.0
_DISTOGRAM_WEIGHT = 0.1
_CLASH_OVERLAP_TOLERANCE = 1.5
_VIOLATION_TOLERANCE_FACTOR = 2.0


def total_loss(config, data: dict, result: dict, *,
               generator: torch.Generator | None = None,
               clash_bernoulli: Tensor | None = None,
               fape_uniform: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute every `default_vp` loss term and their config-weighted sum.

    `data` must supply (all shape `(L, ...)`, flattened multi-item batch):
      `pos` (encoder ground truth, 20-atom), `atom_pos`/`atom_mask` (raw
      atom14 ground truth + empirical mask), `mask`, `aa_gt`, `dssp_gt`,
      `aa_mask`, `dssp_mask`, `t_pos` (the cosine-schedule SIGMA, spec §1.4 -
      NOT raw uniform time), `residue_index`, `chain_index`, `batch_index`.
    `result` must supply the denoiser's output dict: `pos`, `trajectory`,
    `atom_pos`, `aa` (log-probs), `corrupt_aa`, `dssp` (log-probs),
    `distogram` (log-probs), `distogram_neighbours`.

    `fape_uniform`/`clash_bernoulli` let a caller pin the two stochastic draws
    (spec §7.7's fresh Gumbel-top-k, spec §5's `do_clip` gate) for
    reproducible tests; leave `None` for real training.

    Returns `(total, terms)`; `terms` holds every UNWEIGHTED term as
    `losses[...]` would be logged upstream, plus `"total"`.
    """
    c = config
    # Flags that change the loss but are dead for the four released configs.
    # Refuse loudly rather than silently computing a different objective -
    # shipped upstream configs DO set some of these (e.g.
    # default_vp_minimal_timeless_atom14.soft_lddt, default_vp_distogram.
    # distogram_trajectory), so a caller could reasonably hit them.
    _UNIMPLEMENTED = {
        "kabsch": "align ground truth onto the prediction (geometry.align.index_align)",
        "mask_atom14": "per-atom masking; pos_loss currently divides by a fixed atom count",
        "x_late": "restrict the atom14 term to late timesteps",
        "soft_lddt": "soft-LDDT term",
        "aa_trajectory": "per-block sequence loss",
        "distogram_trajectory": "per-block distogram loss",
    }
    unsupported = [k for k in _UNIMPLEMENTED if getattr(c, k, None)]
    if unsupported:
        detail = "; ".join(f"{k}: {_UNIMPLEMENTED[k]}" for k in unsupported)
        raise NotImplementedError(
            f"config sets {unsupported}, which change the loss and are not "
            f"ported (all are dead for the four released checkpoints). {detail}")

    mask = data["mask"].bool()
    batch = data["batch_index"]

    weight = STRUCT.base_weight(mask, batch)
    late = STRUCT.late_mask(data["t_pos"], mask, diffusion_kind=c.diffusion_kind)
    clip = STRUCT.do_clip(batch, getattr(c, "p_clip", 1.0), generator=generator,
                          bernoulli=clash_bernoulli)

    # preconditioning_scale_factors(...)["loss"] == 1 (a bare Python int) for
    # preconditioning=="vp" - the general dispatch is still used so this line
    # does the right thing for edm/edm_scaled/flow configs too (spec §0(b)).
    pos_factors = preconditioning_scale_factors(c, data["t_pos"], c.sigma_data)
    diffusion_weight = weight * pos_factors["loss"]

    terms: dict[str, Tensor] = {}

    # -- sequence-space terms: flat global average (§3), NOT base_weight -----
    terms["aa"] = SEQ.aa_loss(result["aa"], data["aa_gt"], mask, result["corrupt_aa"],
                              data["aa_mask"])
    terms["dssp"] = SEQ.dssp_loss(result["dssp"], data["dssp_gt"], mask, data["dssp_mask"])

    # -- backbone+augment (20-atom) diffusion MSE (§7.4) ----------------------
    terms["pos"] = STRUCT.pos_loss(result["pos"], data["pos"], mask, diffusion_weight, clip)
    terms["pos_trajectory"] = STRUCT.pos_trajectory_loss(
        result["trajectory"], data["pos"], mask, diffusion_weight, clip)

    # -- atom14 diffusion MSE against the EMPIRICAL atom mask (§7.5) --------
    terms["x"] = STRUCT.x_loss(result["atom_pos"], data["atom_pos"], data["atom_mask"],
                               weight, clip)

    # -- backbone rotation (§7.6); gt_frames computed ONCE, reused for the
    #    trajectory variant AND for `local` below (the ground-truth frame
    #    never changes) -------------------------------------------------------
    gt_frames_14, _ = extract_aa_frames(data["atom_pos"])
    terms["rotation"] = STRUCT.rotation_loss(gt_frames_14, result["atom_pos"], mask, weight)
    terms["rotation_trajectory"] = STRUCT.rotation_trajectory_loss(
        gt_frames_14, result["trajectory"], mask, weight)

    # -- sparse SQUARED FAPE (§7.7); `distance`/`pair_mask` reused by `local`
    #    below VERBATIM, `neighbours` here is a FRESH Gumbel-top-k draw,
    #    independent of result["distogram_neighbours"] used by `distogram` --
    distance, pair_mask, frames_gt_20, pos_gt_zeroed = STRUCT.backbone_cb_distance(
        data["pos"], mask, batch)
    neighbours = STRUCT.fape_neighbours(distance, batch, mask, count=c.fape_neighbours,
                                        generator=generator, uniform=fape_uniform)
    per_step = STRUCT.sparse_fape(pos_gt_zeroed, frames_gt_20, result["trajectory"],
                                  neighbours, mask, weight)
    terms["fape"] = STRUCT.fape_loss(per_step)
    terms["fape_trajectory"] = STRUCT.fape_trajectory_loss(per_step)

    # -- all-atom UNSQUARED local FAPE (§7.8), late_mask-gated ---------------
    frames_pred_20, _ = extract_aa_frames(result["trajectory"])
    frames_pred_final = frames_pred_20[-1]                    # == frames of result["pos"]
    terms["local"] = STRUCT.local_loss(
        distance, pair_mask, gt_frames_14, data["atom_pos"], data["atom_mask"],
        frames_pred_final, result["atom_pos"], weight, late, count=c.local_neighbours)

    # -- distogram NLL (§7.9); reuses pos_gt_zeroed's CB (NOT the fape
    #    neighbours - result["distogram_neighbours"] is a separate draw) -----
    terms["distogram"] = SEQ.distogram_loss(
        result["distogram"], result["distogram_neighbours"], pos_gt_zeroed[:, 4], mask, weight)

    # -- AlphaFold violation stack (§7.10), late_mask-gated ------------------
    res_mask = mask & late
    pred_mask = VIOL.get_atom14_mask(data["aa_gt"]) * res_mask[:, None].float()
    violation, _ = VIOL.violation_loss(
        data["aa_gt"], data["residue_index"], result["atom_pos"], pred_mask, res_mask,
        clash_overlap_tolerance=_CLASH_OVERLAP_TOLERANCE,
        violation_tolerance_factor=_VIOLATION_TOLERANCE_FACTOR,
        chain_index=data["chain_index"], batch_index=data["batch_index"], per_residue=False)
    terms["violation"] = violation

    total = (c.aa_weight * terms["aa"]
            + _DSSP_WEIGHT * terms["dssp"]
            + c.pos_weight * terms["pos"]
            + c.trajectory_weight * terms["pos_trajectory"]
            + c.x_weight * terms["x"]
            + c.rotation_weight * terms["rotation"]
            + c.rotation_trajectory_weight * terms["rotation_trajectory"]
            + c.fape_weight * terms["fape"]
            + c.fape_trajectory_weight * terms["fape_trajectory"]
            + c.local_weight * terms["local"]
            + _DISTOGRAM_WEIGHT * terms["distogram"]
            + c.violation_weight * terms["violation"])

    terms["total"] = total
    return total, terms
