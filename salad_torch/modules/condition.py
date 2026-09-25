"""Conditioning inputs: sequence, secondary structure, and pair constraints.

salad's conditioning is *drop-out based*: the network is trained with every
combination of (sequence, per-residue DSSP, per-chain DSSP composition, motif
distance map, chain contacts) present or absent, and each channel is switched
off by a sentinel rather than by a separate code path. So an unconditional
sample is not a different model - it is this module fed
``aa = 20``, ``dssp = 3`` and all-False masks, which makes
`Condition` return exactly ``zeros(L, local_size)``.

Two consequences worth stating plainly:

  * ``one_hot(dssp, 3)`` is evaluated at the sentinel ``3``, i.e. *out of
    range*. JAX returns an all-zero row there and `torch.nn.functional.one_hot`
    raises, which is why everything here goes through `decoders.one_hot`.
  * the per-channel linears and LayerNorms still run in the unconditional case;
    their outputs are discarded by the `torch.where`s. Skipping them would be
    bit-identical (a LayerNorm of a constant vector is finite), but it is not
    worth a device sync, and the modules must exist regardless so the
    checkpoint loads.

Haiku scope: ``structure_diffusion/~prepare_condition/condition``.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..geometry.segment import index_mean
from .basic import HkLayerNorm, HkLinear
from .decoders import one_hot

__all__ = ["Condition", "PrepareCondition", "prepare_condition"]


class Condition(nn.Module):
    """Embeds the three per-residue conditioning channels into ``local_size``."""

    def __init__(self, config, local_size: int | None = None):
        super().__init__()
        c = self.config = config
        size = local_size if local_size is not None else c.local_size
        self.linear_aa = HkLinear(21, size, initializer="glorot")        # linear
        self.layer_norm_aa = HkLayerNorm(size)                           # layer_norm
        self.linear_dssp = HkLinear(3, size, initializer="glorot")       # linear_1
        self.layer_norm_dssp = HkLayerNorm(size)                         # layer_norm_1
        self.linear_dssp_mean = HkLinear(3, size, initializer="glorot")  # linear_2
        self.layer_norm_dssp_mean = HkLayerNorm(size)                    # layer_norm_2

    # ------------------------------------------------------------------
    @staticmethod
    def get_masks(chain_index: Tensor, batch_index: Tensor,
                  set_condition: dict | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Which residues carry which conditioning signal.

        With `set_condition` supplied this is pure bookkeeping over the
        sentinels and consumes no randomness. Upstream also has a training
        branch that *samples* the masks (random spans of sequence and DSSP kept,
        per-chain composition kept with some probability); `prepare_condition`
        always passes `set_condition`, so that branch is unreachable at
        inference and is deliberately not ported.
        """
        if set_condition is None:
            raise NotImplementedError(
                "Condition.get_masks without set_condition is the training-time "
                "random dropout branch; only the inference path is ported.")
        sequence_mask = set_condition["aa"] != 20
        dssp_mask = set_condition["dssp"] != 3
        dssp_mean_mask = set_condition["dssp_mean_mask"]
        return sequence_mask, dssp_mask, dssp_mean_mask

    # ------------------------------------------------------------------
    def forward(self, aa: Tensor, dssp: Tensor, mask: Tensor, residue_index: Tensor,
                chain_index: Tensor, batch_index: Tensor,
                set_condition: dict | None = None
                ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(condition (L, size), sequence_mask, dssp_mask)``.

        `residue_index` is accepted to match upstream's signature and unused.
        """
        sequence_mask, dssp_mask, dssp_mean_mask = self.get_masks(
            chain_index, batch_index, set_condition)

        aa_latent = self.layer_norm_aa(self.linear_aa(one_hot(aa, 21)))
        dssp_latent = self.layer_norm_dssp(self.linear_dssp(one_hot(dssp, 3)))

        # per-chain secondary-structure composition, renormalised to sum to 1
        dssp_mean = index_mean(one_hot(dssp, 3), chain_index, mask[:, None])
        dssp_mean = dssp_mean / torch.clamp(
            dssp_mean.sum(dim=-1, keepdim=True), min=1e-6)
        if set_condition is not None and set_condition.get("dssp_mean") is not None:
            # dead when called from prepare_condition, which always overrides;
            # kept for the training path, which measures it from the target
            dssp_mean = set_condition["dssp_mean"]
        dssp_mean = self.layer_norm_dssp_mean(self.linear_dssp_mean(dssp_mean))

        zero = torch.zeros((), dtype=aa_latent.dtype, device=aa_latent.device)
        condition = (torch.where(dssp_mask[..., None], dssp_latent, zero)
                     + torch.where(dssp_mean_mask[..., None], dssp_mean, zero)
                     + torch.where(sequence_mask[..., None], aa_latent, zero))
        return condition, sequence_mask, dssp_mask


def prepare_condition(condition: Condition, data: dict) -> dict:
    """Build the conditioning inputs from a (possibly empty) request dict.

    Every channel starts at its "unconditional" sentinel and is overridden only
    if `data` carries the corresponding key. The pair channels stay dense
    ``(L, L, ...)`` because `DiffusionPairFeatures` gathers them as
    ``v[index[:, None], neighbours]``.

    Keys read from `data`: ``aa_gt``, ``residue_index``, ``chain_index``,
    ``batch_index``, ``mask``, and optionally ``aa_condition``,
    ``dssp_condition``, ``dssp_mean`` (+ ``dssp_mean_mask``), ``dmap_mask``
    (+ ``dmap``, ``omap``) and ``chain_contacts``.
    """
    aa_gt = data["aa_gt"]
    resi = data["residue_index"]
    chain = data["chain_index"]
    batch = data["batch_index"]
    mask = data["mask"]
    n = aa_gt.shape[0]
    device = aa_gt.device
    float_dtype = torch.float32

    aa = torch.full_like(aa_gt, 20)
    dssp = torch.full_like(aa_gt, 3)
    dssp_mean = torch.zeros((n, 3), dtype=float_dtype, device=device)
    dssp_mean_mask = torch.zeros((n,), dtype=torch.bool, device=device)

    dmap = torch.zeros((n, n), dtype=float_dtype, device=device)
    omap = torch.zeros((n, n, 9), dtype=float_dtype, device=device)
    dmap_mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    chain_contacts = (chain[:, None] != chain[None, :]) & (batch[:, None] == batch[None, :])

    if "aa_condition" in data:
        aa = data["aa_condition"]
    if "dssp_condition" in data:
        dssp = data["dssp_condition"]
    if "dssp_mean" in data:
        value = data["dssp_mean"]
        if value.dim() == 1:
            # a single global composition, applied to every residue
            dssp_mean = torch.stack([value] * n, dim=0)
            dssp_mean_mask = torch.ones((n,), dtype=torch.bool, device=device)
        else:
            dssp_mean = value
            dssp_mean_mask = data["dssp_mean_mask"]
    # the *mask* is the gate, not the map: a dmap without a dmap_mask is ignored
    if "dmap_mask" in data:
        dmap = data["dmap"]
        dmap_mask = data["dmap_mask"]
        if "omap" in data:
            omap = data["omap"]
    if "chain_contacts" in data:
        chain_contacts = data["chain_contacts"]

    # flags channel 0 is "these residues are on different chains"; channels 1-2
    # exist for multi-motif conditioning and stay zero here
    flags = torch.cat((chain_contacts[..., None].to(float_dtype),
                       torch.zeros((n, n, 2), dtype=float_dtype, device=device)),
                      dim=-1)

    set_condition = dict(aa=aa, dssp=dssp, dssp_mean=dssp_mean,
                         dssp_mean_mask=dssp_mean_mask)
    condition_latent, _, _ = condition(aa, dssp, mask, resi, chain, batch,
                                       set_condition=set_condition)
    return {
        "condition": condition_latent,
        "pair_condition": {"dmap": dmap, "omap": omap,
                           "dmap_mask": dmap_mask, "flags": flags},
    }


class PrepareCondition(nn.Module):
    """Owns the `Condition` parameters; Haiku scope ``~prepare_condition``."""

    def __init__(self, config, local_size: int | None = None):
        super().__init__()
        self.condition = Condition(config, local_size)

    def forward(self, data: dict) -> dict:
        return prepare_condition(self.condition, data)
