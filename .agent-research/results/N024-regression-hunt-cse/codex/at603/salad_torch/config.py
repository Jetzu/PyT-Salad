"""Model configurations, ported from `salad.modules.config.noise_schedule_benchmark`.

`DotDict.__getattr__` is `dict.get`, so **any key that is not set reads as
`None`**. That is load-bearing: roughly half the branches in the model are gated
on keys that simply do not exist in `default_vp`, and reproducing the lookup
semantics is what keeps those branches off.

Only the configurations with released checkpoints are defined here. Upstream has
~80 more, almost all experimental derivatives.
"""
from __future__ import annotations

from copy import deepcopy

__all__ = ["DotDict", "default_vp", "default_vp_scaled", "default_ve_scaled",
           "multimotif_vp", "CONFIGS", "get_config"]


class DotDict(dict):
    """Attribute access over a dict, where a missing key is `None`, not an error."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self) -> "DotDict":
        return DotDict(deepcopy(dict(self)))


default_vp = DotDict(
    # encoder (unused for de-novo generation, but its weights are in the file)
    augment_size=15,
    encoder_depth=2,
    # feature sizes
    local_size=128,
    pair_size=64,
    num_neighbours=48,
    # module parameters
    resi_dual=False,
    relative_position_encoding_max=32,   # never read; relpos count is hardcoded 32
    factor=4,
    heads=8,
    key_size=32,
    multi_query=False,
    # diffusion stack
    diffusion_depth=6,
    block_size=1,
    diffusion_kind="vpfixed",
    pos_mean_sigma=1.6,
    pos_std_sigma=1.4,
    sigma_data=10.0,
    preconditioning="vp",
    diffusion_time_scale="cosine",
    # sequence
    diffuse_sequence=False,
    aa_decoder_kind="adm",
    aa_decoder_depth=3,
    # loss
    p_clip=1.0,
    p_drop_dssp=0.2,
    time_embedding=True,
    local_neighbours=16,
    fape_neighbours=64,
    pos_weight=2.0,
    trajectory_weight=1.0,
    x_weight=2.0,
    rotation_weight=2.0,
    rotation_trajectory_weight=1.0,
    violation_weight=0.1,
    local_weight=10.0,
    aa_weight=10.0,
    fape_weight=1,
    fape_trajectory_weight=0.5,
    # dataset constraints (read by the data pipeline, not the model)
    min_size=50,
    max_size=None,
)

#: same parameter tree as `default_vp`; only the noising differs (the cloud std
#: becomes the per-structure CA spread instead of a fixed sigma_data)
default_vp_scaled = default_vp.copy()
default_vp_scaled.diffusion_kind = "vp"

#: VE / EDM: adds input+skip scaling and feeds `init_pos` to every pair block,
#: which is why its checkpoint has two extra linears per pair block
default_ve_scaled = default_vp.copy()
default_ve_scaled.preconditioning = "edm_scaled"
default_ve_scaled.diffusion_kind = "edm"
default_ve_scaled.pos_weight *= 100
default_ve_scaled.trajectory_weight *= 100

#: multi-motif scaffolding: MotifAdapterDiffusionBlock + a dmap_mask embedding
multimotif_vp = default_vp.copy()
multimotif_vp.multi_motif = True

CONFIGS = {
    "default_vp": default_vp,
    "default_vp_scaled": default_vp_scaled,
    "default_ve_scaled": default_ve_scaled,
    "multimotif_vp": multimotif_vp,
}


def get_config(name: str) -> DotDict:
    """Fetch a named config by value, so callers can mutate it safely."""
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; have {sorted(CONFIGS)}")
    return CONFIGS[name].copy()


def uses_init_pos(config) -> bool:
    """Whether each pair block gets the extra `init_pos` distance+rotation pair.

    Two independent routes switch it on: `EDMDiffusionStack` supplies `init_pos`
    for edm_scaled, and `MotifAdapterDiffusionBlock` passes `init_pos=prev_pos`
    unconditionally. Both change the flat `linear_N` numbering, so the
    checkpoint converter needs this too.
    """
    return config.preconditioning == "edm_scaled" or bool(config.multi_motif)
