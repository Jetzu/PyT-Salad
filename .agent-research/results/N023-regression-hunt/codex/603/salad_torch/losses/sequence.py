"""Sequence-space losses: `aa` (ADM NLL), `dssp` (secondary-structure NLL),
`distogram` (pairwise distance NLL).

Ported from `salad.modules.noise_schedule_benchmark.loss()` per
`notes/specs/03_training_losses.md` §7.1, §7.3, §7.9, for `default_vp`.

`aa_loss`/`dssp_loss` use a FLAT GLOBAL average over every valid, predictable
residue in the whole flattened batch - NOT `losses.structure.base_weight`'s
per-item-equalised reduction. `distogram_loss` breaks that pattern and DOES
use `base_weight`, despite superficially looking like "another NLL over a
fixed neighbour set" like `aa`/`dssp`. All three asymmetries are real,
present in the reference, and intentional (spec §3) - do not "fix" them into
a single consistent reduction.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..geometry.features import distance_one_hot
from ..geometry.rigid import vec_norm

__all__ = ["aa_loss", "dssp_loss", "distogram_loss"]


def _one_hot_or_zero(x: Tensor, n: int) -> Tensor:
    """`jax.nn.one_hot` semantics: an out-of-range index yields an all-zero
    row (torch's `F.one_hot` raises instead). Needed because `aa_gt == 20`
    (unresolved identity) must not error when scored against a 20-class
    distribution - it must simply contribute 0 to the NLL sum, which this
    produces automatically since the "target" row is all zero.
    """
    return (x[..., None] == torch.arange(n, device=x.device)).to(torch.float32)


# ---------------------------------------------------------------------------
# aa (spec §7.1) - ADM-style amino-acid NLL
# ---------------------------------------------------------------------------
def aa_loss(aa_log_probs: Tensor, aa_gt: Tensor, mask: Tensor, corrupt_aa: Tensor,
           aa_mask: Tensor) -> Tensor:
    """spec §7.1 `aa`. FLAT GLOBAL average (§3) - an asymmetry against every
    structural term in `losses.structure`, reproduced deliberately.

    `corrupt_aa` MUST be `result["corrupt_aa"]` - the ADM `AADecoder`'s OWN,
    independently-sampled per-residue masking draw (already further masked by
    `data["mask"] & (aa_gt != 20)` inside the model, `Diffusion.forward`) -
    NEVER `data["t_seq"]`/`seq_noised`. For `default_vp`, `c.diffuse_sequence`
    is False: the trunk never sees noised (or clean) sequence at all, and the
    entire amino-acid task is delegated to this self-contained decoder, whose
    masking ratio has nothing to do with `data["t_seq"]`.

    `aa_log_probs`: `(L, 20)` log-softmax (cannot predict the mask token
    itself). `aa_mask`: True where the identity was given as a hard condition
    - excluded here, the model isn't asked to predict what it was told.
    """
    predict_mask = mask.bool() & (aa_gt != 20) & corrupt_aa.bool() & ~aa_mask.bool()
    nll = -(aa_log_probs * _one_hot_or_zero(aa_gt, 20)).sum(-1)
    nll = torch.where(predict_mask, nll, torch.zeros_like(nll))
    return nll.sum() / predict_mask.sum().clamp(min=1).float()


# ---------------------------------------------------------------------------
# dssp (spec §7.3) - secondary-structure NLL, bare linear probe
# ---------------------------------------------------------------------------
def dssp_loss(dssp_log_probs: Tensor, dssp_gt: Tensor, mask: Tensor,
             dssp_mask: Tensor) -> Tensor:
    """spec §7.3 `dssp`. Same flat-global-average pattern as `aa_loss`.
    `dssp_log_probs`: `(L, 3)` log-softmax of a bare `Linear(3, bias=False)`
    on the trunk's final `local` - no MLP, no decoder stack. `dssp_mask`:
    True where secondary structure was given as a hard condition - excluded.
    """
    predict_mask = mask.bool() & ~dssp_mask.bool()
    nll = -(dssp_log_probs * _one_hot_or_zero(dssp_gt, 3)).sum(-1)
    nll = torch.where(predict_mask, nll, torch.zeros_like(nll))
    return nll.sum() / predict_mask.sum().clamp(min=1).float()


# ---------------------------------------------------------------------------
# distogram (spec §7.9) - 64-bin pairwise distance NLL, base_weight reduction
# ---------------------------------------------------------------------------
def distogram_loss(distogram_log_probs: Tensor, neighbours: Tensor, cb: Tensor,
                   mask: Tensor, weight: Tensor) -> Tensor:
    """spec §7.9 `distogram`. `neighbours` MUST be `result["distogram_neighbours"]`
    - a `(L, 64)` Gumbel-top-k draw computed ONCE, earlier in the forward pass,
    off the ENCODER's CB (`data["pos"][:, 4]`). This is a SEPARATE RNG draw
    from `losses.structure.fape_neighbours` (same count, same selector,
    different call site) - never conflate the two neighbour sets.

    `cb` must be the SAME zeroed-at-invalid-residues CB column used by the
    fape/local setup (`losses.structure.backbone_cb_distance(...)[3][:, 4]`),
    not a freshly sliced `data["pos"][:, 4]` - reproduced for parity even
    though the difference is only visible at invalid residues (which are
    masked out downstream regardless).

    `distogram_log_probs`: `(L, 64, 64)` log-softmax over 64 distance bins
    for each of 64 neighbours.
    """
    pair_mask = (neighbours != -1) & mask[:, None].bool() & mask[neighbours].bool()
    distance_gt = vec_norm(cb[:, None] - cb[neighbours])
    target = distance_one_hot(distance_gt).detach()             # stop_gradient, 64 bins over [0, 22]
    nll = -(distogram_log_probs * target).sum(-1)                # (L, 64)
    nll = torch.where(pair_mask, nll, torch.zeros_like(nll))
    per_residue = nll.sum(-1) / pair_mask.sum(-1).clamp(min=1).float()
    return (per_residue * weight).sum()
