"""Structural diffusion losses: `pos`/`x`/`rotation`/`fape`/`local`.

Ported from `salad.modules.noise_schedule_benchmark.loss()` per
`notes/specs/03_training_losses.md` §3-4, §7.4-§7.8, for `default_vp`.

Every term here reduces with `base_weight` (§3): summing `base_weight` over a
batch's residues gives an UNWEIGHTED MEAN OVER BATCH ITEMS, so a long protein
and a short one in the same batch contribute equally. Contrast
`losses.sequence.aa_loss`/`dssp_loss`, which instead do a FLAT GLOBAL average
over every valid residue - longer proteins dominate those two terms
proportionally to their length. That asymmetry is real, present in the
reference, and intentional (spec §3); it is reproduced here on purpose, not
an oversight to "fix" into consistency.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..geometry.frames import extract_aa_frames
from ..geometry.neighbours import get_neighbours, get_random_neighbours
from ..geometry.rigid import Rigid, vec_norm, vec_norm2
from ..geometry.segment import index_sum

__all__ = [
    "base_weight", "late_mask", "do_clip",
    "pos_loss", "pos_trajectory_loss",
    "x_loss",
    "rotation_loss", "rotation_trajectory_loss",
    "backbone_cb_distance", "fape_neighbours", "sparse_fape",
    "fape_loss", "fape_trajectory_loss",
    "local_loss",
]


# ---------------------------------------------------------------------------
# shared reduction / gating primitives (spec §2, §3, §5)
# ---------------------------------------------------------------------------
def base_weight(mask: Tensor, batch: Tensor) -> Tensor:
    """`get_batch_weight`, spec §3: `mask / item_size / num_items`.

    Summing this over all `L` residues in a loss term yields an unweighted
    mean over batch items - every protein counts equally regardless of length.
    """
    mask_f = mask.float()
    count = index_sum(mask_f, batch, mask)  # per-item valid-residue count, broadcast back
    # keep this as a 0-d tensor: .item() forces a device->host sync every
    # step and breaks torch.compile graphs. JAX keeps it traced.
    num_items = (batch.max() + 1).to(mask_f.dtype) if batch.numel() else 1.0
    return mask_f / count.clamp(min=1.0) / num_items


def late_mask(t_pos: Tensor, mask: Tensor, diffusion_kind: str = "vpfixed") -> Tensor:
    """spec §2. `t_pos` here is the cosine-schedule SIGMA output (§1.4), not
    raw uniform time - so "late" means LOW noise (`sigma < 0.5`), the tail of
    the reverse trajectory close to the clean structure. Used by `local_loss`
    and the `violation` term (assembled in `losses.total`).
    """
    threshold = 5.0 if diffusion_kind == "edm" else 0.5
    return (t_pos < threshold) & mask.bool()


def do_clip(batch: Tensor, p_clip: float = 1.0, *, generator: torch.Generator | None = None,
           bernoulli: Tensor | None = None) -> Tensor:
    """spec §5 `p_clip`/`do_clip`: a per-residue squared-distance clip bound.

    The reference draws one bool per element of `batch.shape` (i.e. up to `L`
    of them, more than the actual number of distinct items) and only THEN
    indexes it by `batch`, rather than drawing exactly one bool per item;
    reproduced exactly so the RNG-shape idiom is visible even though torch and
    JAX do not share a random stream. For `p_clip=1.0` (`default_vp`) the gate
    never actually fires and this is the constant 100.0 everywhere.
    """
    n = batch.shape[0]
    if bernoulli is None:
        bernoulli = torch.rand(n, generator=generator, device=batch.device) < p_clip
    gate = bernoulli[batch]
    hundred = torch.full((n,), 100.0, device=batch.device, dtype=torch.float32)
    inf = torch.full((n,), float("inf"), device=batch.device, dtype=torch.float32)
    return torch.where(gate, hundred, inf)


# ---------------------------------------------------------------------------
# pos / pos_trajectory (spec §7.4) - backbone+augment (20-atom) diffusion MSE
# ---------------------------------------------------------------------------
def pos_loss(pos_pred: Tensor, pos_gt: Tensor, mask: Tensor, weight: Tensor,
            clip: Tensor) -> Tensor:
    """spec §7.4 `pos`. `pos_pred`/`pos_gt`: `(L, A=20, 3)`.

    `weight` is `base_weight * diffusion_weight["loss"]`; for `default_vp` the
    preconditioning factor is the literal constant 1 (spec §0(b)), so callers
    may simply pass `base_weight` unmodified and get bit-identical behaviour.
    There is no per-atom validity mask over the 20 encoder-space atoms (every
    slot is "present" once the residue itself is valid - the encoder invented
    all 20), so the atom-count denominator is the fixed `A`, not a per-residue
    count.
    """
    dist2 = vec_norm2(pos_pred - pos_gt)                     # (L, A), no eps - plain squared error
    dist2 = torch.minimum(dist2, clip[:, None])              # per-atom clip to 100.0 (do_clip)
    n_atoms = pos_gt.shape[-2]
    per_residue = dist2.sum(-1) / n_atoms
    per_residue = torch.where(mask.bool(), per_residue, torch.zeros_like(per_residue))
    return (per_residue * weight).sum() / 3.0                # /3: sum-of-3-sq-errors -> per-coord MSE


def pos_trajectory_loss(trajectory_pred: Tensor, pos_gt: Tensor, mask: Tensor,
                        weight: Tensor, clip: Tensor) -> Tensor:
    """spec §7.4 `pos_trajectory`. Identical formula to `pos_loss`, broadcast
    over the leading `D` trajectory axis via ordinary elementwise ops (no vmap
    needed), then a PLAIN, UNWEIGHTED mean over `D` - contrast the one
    exception, the dead `aa_trajectory`, which uses a `0.99**step` decay.
    """
    dist2 = vec_norm2(trajectory_pred - pos_gt[None])        # (D, L, A)
    dist2 = torch.minimum(dist2, clip[None, :, None])
    n_atoms = pos_gt.shape[-2]
    per_residue = dist2.sum(-1) / n_atoms                    # (D, L)
    per_residue = torch.where(mask[None].bool(), per_residue, torch.zeros_like(per_residue))
    per_step = (per_residue * weight[None]).sum(-1) / 3.0    # (D,)
    return per_step.mean()


# ---------------------------------------------------------------------------
# x (spec §7.5) - atom14 diffusion MSE against the EMPIRICAL atom mask
# ---------------------------------------------------------------------------
def x_loss(atom_pos_pred: Tensor, atom_pos_gt: Tensor, atom_mask: Tensor,
          weight: Tensor, clip: Tensor) -> Tensor:
    """spec §7.5 `x`. `atom_mask` MUST be `data["atom_mask"]` (whether the
    crystal structure resolved that atom) - NOT the canonical per-restype mask
    used by `losses.violation` (`get_atom14_mask`). These usually agree but
    are conceptually different masks. `clip` is the SAME `do_clip` draw used
    by `pos_loss`/`pos_trajectory_loss` - it is not resampled here. Unlike
    `pos_loss`, `weight` here is plain `base_weight` with no preconditioning
    factor at all (spec §7.5 never mentions `diffusion_weight`).
    """
    dist2 = vec_norm2(atom_pos_gt - atom_pos_pred)            # (L, 14)
    dist2 = torch.minimum(dist2, clip[:, None])
    m = atom_mask.bool()
    summed = torch.where(m, dist2, torch.zeros_like(dist2)).sum(-1) / 3.0
    per_residue = summed / m.sum(-1).clamp(min=1).float()
    return (per_residue * weight).sum()


# ---------------------------------------------------------------------------
# rotation / rotation_trajectory (spec §7.6) - frames from ATOM14 positions
# ---------------------------------------------------------------------------
def rotation_loss(gt_frames: Rigid, atom_pos_pred: Tensor, mask: Tensor,
                  weight: Tensor) -> Tensor:
    """spec §7.6 `rotation`. `gt_frames` must be `extract_aa_frames(atom_pos_gt)`
    (atom14 ground truth) - callers compute it ONCE and pass it here AND to
    `rotation_trajectory_loss`, since the ground-truth frame never changes
    over the trajectory (mirrors the reference's reuse pattern exactly).
    `extract_aa_frames` only reads slots 0/1/2 (N, CA, C) regardless of the
    tensor's total atom count, so this is correct whether given atom14 or the
    20-atom backbone+augment tensor - but the reference always starts from
    atom14 here, so keep that convention for parity/debugging.
    """
    frames, _ = extract_aa_frames(atom_pos_pred)
    rel = gt_frames.rotation.transpose(-1, -2) @ frames.rotation   # R_gt^T @ R_pred
    eye = torch.eye(3, device=rel.device, dtype=rel.dtype)
    loss = ((rel - eye) ** 2).sum((-1, -2))                        # squared Frobenius deviation, no /3
    loss = torch.where(mask.bool(), loss, torch.zeros_like(loss))
    return (loss * weight).sum()


def rotation_trajectory_loss(gt_frames: Rigid, trajectory_pred: Tensor, mask: Tensor,
                             weight: Tensor) -> Tensor:
    """spec §7.6 `rotation_trajectory`. `trajectory_pred` is the 20-atom
    `result["trajectory"]`, not atom14 - only its slots 0/1/2 are read, which
    coincide with atom14's slots 0-3 by construction (spec §7.5), so this is
    numerically consistent with `rotation_loss` at the final step.
    `gt_frames` is the SAME object passed to `rotation_loss` - it is broadcast
    over `D`, never recomputed per step.
    """
    frames, _ = extract_aa_frames(trajectory_pred)                 # (D, L) Rigid, plain broadcasting
    rel = gt_frames.rotation[None].transpose(-1, -2) @ frames.rotation
    eye = torch.eye(3, device=rel.device, dtype=rel.dtype)
    loss = ((rel - eye) ** 2).sum((-1, -2))                        # (D, L)
    loss = torch.where(mask[None].bool(), loss, torch.zeros_like(loss))
    per_step = (loss * weight[None]).sum(-1)                       # (D,)
    return per_step.mean()                                         # plain mean, no decay


# ---------------------------------------------------------------------------
# fape / fape_trajectory (spec §7.7) - sparse SQUARED FAPE, clip=100.0
# ---------------------------------------------------------------------------
def backbone_cb_distance(pos_gt: Tensor, mask: Tensor,
                         batch: Tensor) -> tuple[Tensor, Tensor, Rigid, Tensor]:
    """Shared setup for `fape`/`fape_trajectory`/`local_loss` (spec §7.7-7.8):
    the 20-atom-space CB-CB distance matrix, its pair mask, the ground-truth
    frames, and `pos_gt` with invalid residues zeroed. `local_loss` reuses the
    `distance`/`pair_mask` returned here VERBATIM - do not recompute them.
    """
    pair_mask = (batch[:, None] == batch[None, :]) & mask[:, None].bool() & mask[None, :].bool()
    pos_gt_zeroed = torch.where(mask[:, None, None].bool(), pos_gt, torch.zeros_like(pos_gt))
    # explicit stop_gradient on the ground truth (spec §7.7): defensive/no-op
    # in practice since GT carries no grad, replicated here for fidelity.
    frames_gt, _ = extract_aa_frames(pos_gt_zeroed.detach())
    cb = pos_gt_zeroed[:, 4]                                       # slot 4 = idealised CB
    distance = vec_norm(cb[:, None] - cb[None, :])
    distance = torch.where(pair_mask, distance, torch.full_like(distance, float("inf")))
    return distance, pair_mask, frames_gt, pos_gt_zeroed


def fape_neighbours(distance: Tensor, batch: Tensor, mask: Tensor, count: int = 64,
                    generator: torch.Generator | None = None,
                    uniform: Tensor | None = None) -> Tensor:
    """A FRESH Gumbel-top-k draw for sparse FAPE - INDEPENDENT of
    `result["distogram_neighbours"]` consumed by `losses.sequence.distogram_loss`.
    Same count (64) and same selector as the distogram draw, but a SEPARATE
    RNG call at a different point in the pipeline. Never conflate the two.
    """
    return get_random_neighbours(count)(distance, batch, mask, generator=generator,
                                        uniform=uniform)


def sparse_fape(pos_gt_zeroed: Tensor, frames_gt: Rigid, trajectory_pred: Tensor,
                neighbours: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
    """Per-step SQUARED sparse FAPE (spec §7.7), clipped at 100.0 Ų, reduced
    with `weight` (`base_weight`) over residues - `(D,)`. NOT yet divided by 3
    or scaled by `fape_weight`/`fape_trajectory_weight`; `fape_loss`/
    `fape_trajectory_loss` below finish the reduction (see their docstrings
    for why applying `/3` per-term here is numerically identical to the
    reference's single `/3` on the combined weighted sum).
    """
    valid = (neighbours != -1)
    mask_neighbours = valid & mask[:, None].bool() & mask[neighbours].bool()   # (L, K)

    gt_local = frames_gt[:, None, None].apply_inverse_to_point(pos_gt_zeroed[neighbours])
    frames_pred, _ = extract_aa_frames(trajectory_pred)                        # (D, L)
    pred_local = frames_pred[:, :, None, None].apply_inverse_to_point(
        trajectory_pred[:, neighbours])                                       # (D, L, K, A, 3)

    sq_err = torch.clamp(vec_norm2(pred_local - gt_local[None]), max=100.0)    # (D, L, K, A)
    sq_err = sq_err.mean(-1)                                                   # mean over 20 atoms -> (D, L, K)
    sq_err = torch.where(mask_neighbours[None], sq_err, torch.zeros_like(sq_err))
    denom = mask_neighbours.sum(-1).clamp(min=1).float()                       # (L,)
    per_residue = sq_err.sum(-1) / denom[None]                                 # (D, L)
    per_residue = torch.where(mask[None].bool(), per_residue, torch.zeros_like(per_residue))
    return (per_residue * weight[None]).sum(-1)                                # (D,)


def fape_loss(per_step: Tensor) -> Tensor:
    """spec §7.7 `fape`: only the FINAL trajectory step, `/3` applied (the
    per-coordinate MSE conversion, spec §7.4)."""
    return per_step[-1] / 3.0


def fape_trajectory_loss(per_step: Tensor) -> Tensor:
    """spec §7.7 `fape_trajectory`: a PLAIN mean over all `D` steps, `/3`
    applied. Combining `fape_weight*fape_loss(...) + fape_trajectory_weight*
    fape_trajectory_loss(...)` in `losses.total` gives EXACTLY the reference's
    `(fape_weight*per_step[-1] + fape_trajectory_weight*mean(per_step)) / 3`
    (NSB:1334) - the two orderings of "divide by 3" and "weighted-sum" commute
    because both are linear, so no extra `/3` is needed at the combination site.
    """
    return per_step.mean() / 3.0


# ---------------------------------------------------------------------------
# local (spec §7.8) - all-atom UNSQUARED local-neighbourhood FAPE, clip=10.0
# ---------------------------------------------------------------------------
def local_loss(distance: Tensor, pair_mask: Tensor, gt_frames_14: Rigid,
              atom_pos_gt: Tensor, atom_mask: Tensor, frames_pred_final: Rigid,
              atom_pos_pred: Tensor, weight: Tensor, gate: Tensor,
              count: int = 16) -> Tensor:
    """spec §7.8 `local`. Reuses `distance`/`pair_mask` from
    `backbone_cb_distance` (20-atom-space CB matrix) VERBATIM for a
    DETERMINISTIC k-NN (`get_neighbours`, not a Gumbel draw). `gt_frames_14`
    is `losses.structure.rotation_loss`'s `gt_frames` argument (ATOM14 ground
    truth, extract_aa_frames(atom_pos_gt)) - a DIFFERENT `Rigid` object from
    `backbone_cb_distance`'s 20-atom-space `frames_gt`, reused from §7.6, not
    recomputed. `frames_pred_final` is the LAST trajectory step's predicted
    frames (== `extract_aa_frames(result["pos"])`, since `trajectory[-1] ==
    result["pos"]`); pass `losses.structure.sparse_fape`'s
    `extract_aa_frames(trajectory_pred)[-1]` to avoid recomputing it.

    THE KEY TRAP (flagged explicitly, spec §7.8): the reference names this
    quantity `dist2` but it is **UNSQUARED** (`.norm()`, not `.norm2()`) and
    clipped at **10.0** (a raw Å bound, not the 100.0 Ų bound `fape`/`pos`/`x`
    use) - do not square it when porting, despite the variable name.
    """
    neighbours = get_neighbours(count)(distance, pair_mask)
    valid = (neighbours != -1)
    mask_neighbours = valid[..., None] & atom_mask[neighbours].bool()          # (L, K, 14)

    gt_local = gt_frames_14[:, None, None].apply_inverse_to_point(atom_pos_gt[neighbours])
    pred_local = frames_pred_final[:, None, None].apply_inverse_to_point(
        atom_pos_pred[neighbours])

    dist = vec_norm(pred_local - gt_local)                                     # UNSQUARED - see docstring
    dist = torch.clamp(dist, max=10.0)                                         # clip the raw distance
    dist = torch.where(mask_neighbours, dist, torch.zeros_like(dist))
    summed = dist.sum(dim=(-1, -2)) / 3.0                                      # jointly over (neighbour, atom)
    denom = mask_neighbours.sum(dim=(-1, -2)).clamp(min=1).float()
    per_residue = summed / denom
    # item-equalised AND late_mask-gated - NOT renormalised for the reduced
    # item count when late_mask drops a whole batch item (spec §7.8).
    return (per_residue * weight * gate.float()).sum()
