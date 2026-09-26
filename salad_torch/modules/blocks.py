"""Pair features, feature updates and the diffusion block/stack.

Submodules here use *semantic* names (``linear_relpos``, ``linear_dmap``, ...)
grouped into `DiffusionPairFeatures` / `Update` objects. Haiku does NOT group
them: `diffusion_pair_features` and `update_positions` are plain functions, so
their layers share the enclosing block's name counter and land flat in the
``diffusion_block`` scope as ``linear`` .. ``linear_14``.

Conversion is therefore an explicit table, not a name identity - see
`salad_torch/convert/`. The bijection is shape-exact (verified: all 6 layers of
``default_vp-200k`` load with ``strict=True``, 576 tensors, 0 missing).

The flat indices are config-dependent: ``multi_motif`` inserts a dmap_mask
linear early, and an ``init_pos`` branch adds two more per pair block, so
"``linear_14`` is the position update" holds only for ``default_vp``
(``default_ve_scaled`` -> ``linear_18``, ``multimotif_vp`` -> ``linear_20``).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..diffusion.precond import edm_scaling
from ..geometry import features as GF
from ..geometry.frames import aa_frames_t, apply_t
from ..geometry.neighbours import (get_index_neighbours, get_neighbours,
                                   get_random_neighbours)
from ..geometry.segment import index_mean
from .. import opt
from .attention import SparseStructureAttention
from .basic import MLP, HkLayerNorm, HkLinear, gelu

__all__ = ["DiffusionPairFeatures", "Update", "EncoderUpdate", "UpdatePositions",
           "DiffusionBlock", "DiffusionStack", "extract_condition_neighbours",
           "condition_neighbour_prologue", "condition_neighbour_tail",
           "MotifAdapterDiffusionBlock", "extract_motif_neighbours"]


def condition_neighbour_prologue(num_index: int, num_spatial: int):
    """The part of `extract_condition_neighbours` that is loop-INVARIANT (P6).

    The self-conditioning branch ranks against ``prev_pos``, which the trunk
    never mutates, so its distance matrix, its sequence-index stage and its
    spatial stage are the same tensors in all six blocks. Building them once
    and reusing them is common-subexpression elimination: bit-exact, because
    the removed work is literally the same arithmetic re-executed.

    Returns ``(distance, seed_neighbours)`` for `condition_neighbour_tail`.
    """

    def inner(pos: Tensor, pair_condition: dict | None, resi: Tensor,
              chain: Tensor, batch: Tensor, mask: Tensor,
              index_neighbours: Tensor | None = None) -> tuple[Tensor, Tensor]:
        neighbours = (get_index_neighbours(num_index)(resi, chain, batch, mask)
                      if index_neighbours is None else index_neighbours)
        ca = pos[:, 1]
        dist = torch.sqrt(torch.clamp(
            ((ca[:, None] - ca[None, :]) ** 2).sum(-1), min=1e-12))
        if pair_condition is not None:
            # P4: with no pair conditioning `dmap_mask` is all-false, so this
            # `where` is the identity on `dist`. The caller passes None to say
            # so; it never guesses.
            dist = torch.where(pair_condition["dmap_mask"],
                               pair_condition["dmap"], dist)
        same_batch = batch[:, None] == batch[None, :]
        pair_mask = mask[:, None] & mask[None, :] & same_batch
        return dist, get_neighbours(num_spatial)(dist, pair_mask, neighbours)

    return inner


def condition_neighbour_tail(num_random: int, num_block: int):
    """The per-block part: the Gumbel stage, and the block-contact stage.

    ``num_block=0`` drops the fourth stage entirely (P5) rather than calling
    ``get_random_neighbours(0)``, which would still build a whole ``(N, N)``
    Gumbel field before concatenating nothing. That is only legal when
    ``flags`` is all-false, in which case the stage's distance matrix is
    ``+inf`` everywhere and every one of its 16 columns is the ``-1``
    sentinel - see the caller's runtime gate.
    """

    def inner(dist: Tensor, seed: Tensor, pair_condition: dict | None,
              batch: Tensor, mask: Tensor, uniform: Tensor | None = None,
              uniform2: Tensor | None = None) -> Tensor:
        neighbours = get_random_neighbours(num_random)(
            dist, batch, mask, seed, uniform=uniform)
        if num_block <= 0:
            return neighbours
        flags_any = pair_condition["flags"].any(dim=-1)
        block_contact_dist = torch.where(
            flags_any, flags_any.to(dist.dtype),
            torch.full_like(dist, float("inf")))
        return get_random_neighbours(num_block)(
            block_contact_dist, batch, mask, neighbours,
            uniform=uniform2 if uniform2 is not None else uniform)

    return inner


def extract_condition_neighbours(num_index: int, num_spatial: int,
                                 num_random: int, num_block: int):
    """Neighbours for the self-conditioning branch.

    Motif distances override the geometric CA distance where ``dmap_mask`` is
    set, and a final stage samples among residues flagged by pair conditioning.

    This is exactly ``condition_neighbour_prologue`` followed by
    ``condition_neighbour_tail``; the two halves exist separately so the
    invariant half can be hoisted out of the 6-block loop (P6).
    """
    prologue = condition_neighbour_prologue(num_index, num_spatial)
    tail = condition_neighbour_tail(num_random, num_block)

    def inner(pos: Tensor, pair_condition: dict, resi: Tensor, chain: Tensor,
              batch: Tensor, mask: Tensor, uniform: Tensor | None = None,
              uniform2: Tensor | None = None) -> Tensor:
        dist, seed = prologue(pos, pair_condition, resi, chain, batch, mask)
        return tail(dist, seed, pair_condition, batch, mask, uniform, uniform2)

    return inner


class DiffusionPairFeatures(nn.Module):
    """`diffusion_pair_features` - the production pair-feature builder."""

    def __init__(self, config, atom_count: int, use_init_pos: bool = False):
        super().__init__()
        c = self.config = config
        p = c.pair_size
        self.use_init_pos = use_init_pos
        self.multi_motif = bool(c.multi_motif)
        n_dir = atom_count * 3
        n_pairvec = 12 * atom_count

        # creation order below must match Haiku's, it fixes the linear_N indices
        self.linear_relpos = HkLinear(66, p, bias=False)
        if self.multi_motif:
            self.linear_dmap_mask = HkLinear(1, p, bias=True)
        self.linear_dmap = HkLinear(64, p, bias=False)
        self.linear_flags = HkLinear(3, p, bias=False)
        self.linear_dist = HkLinear(400, p, bias=False)
        self.linear_dir = HkLinear(n_dir, p, bias=False)
        self.linear_rot = HkLinear(9, p, bias=False)
        self.linear_pairvec = HkLinear(n_pairvec, p, bias=False)
        if use_init_pos:
            self.linear_init_dist = HkLinear(400, p, bias=False)
            self.linear_init_rot = HkLinear(9, p, bias=False)
        self.layer_norm = HkLayerNorm(p)
        self.mlp = MLP(p, p * 2, out_size=p, depth=2, activation=gelu,
                       final_init="linear")

    def forward(self, pos: Tensor, pair_condition: dict | None, neighbours: Tensor,
                resi: Tensor, chain: Tensor, batch: Tensor, mask: Tensor,
                cyclic_mask: Tensor | None = None,
                init_pos: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # A checkpoint that has the init_pos weights must always be given
        # init_pos: otherwise two loaded matrices are silently skipped and the
        # output is quietly wrong. Fail loudly instead.
        if self.use_init_pos and init_pos is None:
            raise ValueError(
                "this block was built with use_init_pos=True (its checkpoint has "
                "init_pos weights) but init_pos was not supplied")
        mask = mask.bool()
        pair_mask = mask[:, None] & mask[neighbours] & (neighbours != -1)
        n = pos.shape[0]
        index = torch.arange(n, device=pos.device)

        if pair_condition is not None:
            # Only the three keys `forward` actually consumes. `omap` is the
            # fourth entry of `pair_condition` and it is read by NOTHING here:
            # the orientation embedding is `linear_omap`, which only exists
            # when `config.use_omap` is set, and no released checkpoint sets
            # it (see convert/haiku_to_torch._pair_feature_names). Gathering
            # it anyway cost one dead (N, K, 9) gather per pair block, twelve
            # per forward. Bit-exact to drop: the result was discarded.
            pc = {k: pair_condition[k][index[:, None], neighbours]
                  for k in ("dmap", "dmap_mask", "flags")}
        else:
            pc = None

        # P3: the relative-position term is a row gather written as a
        # (N, K, 66) one-hot times a (66, 64) matrix. `forward_indexed` is the
        # same bits (see HkLinear.forward_indexed); the bins are structurally
        # in [0, 65] because `sequence_relative_position` clamps them.
        if opt.GATHER_ONEHOT:
            relpos = GF.sequence_relative_position(
                32, one_hot=False, cyclic=cyclic_mask is not None)(
                    resi, chain, batch, neighbours, cyclic_mask=cyclic_mask)
            pair = self.linear_relpos.forward_indexed(relpos.long())
        else:
            relpos = GF.sequence_relative_position(
                32, one_hot=True, cyclic=cyclic_mask is not None)(
                    resi, chain, batch, neighbours, cyclic_mask=cyclic_mask)
            pair = self.linear_relpos(relpos)

        if pc is not None:
            if self.multi_motif:
                pair = pair + self.linear_dmap_mask(pc["dmap_mask"][..., None].float())
            dmap_rbf = torch.where(pc["dmap_mask"][..., None],
                                   GF.distance_rbf(pc["dmap"]),
                                   torch.zeros((), device=pos.device, dtype=pair.dtype))
            pair = pair + self.linear_dmap(dmap_rbf)
            flags = torch.where(pc["flags"].any(dim=-1)[..., None],
                                pc["flags"], torch.zeros_like(pc["flags"]))
            pair = pair + self.linear_flags(flags)

        pair = pair + self.linear_dist(
            GF.distance_features(pos, neighbours, 0.0, 22.0))
        pair = pair + self.linear_dir(GF.direction_features(pos, neighbours))
        pair = pair + self.linear_rot(GF.position_rotation_features(pos, neighbours))
        pair = pair + self.linear_pairvec(GF.pair_vector_features(pos, neighbours))

        if self.use_init_pos:
            pair = pair + self.linear_init_dist(
                GF.distance_features(init_pos, neighbours, 0.0, 22.0))
            pair = pair + self.linear_init_rot(
                GF.position_rotation_features(init_pos, neighbours))

        pair = self.layer_norm(pair)
        pair = self.mlp(pair)
        return pair, pair_mask


class Update(nn.Module):
    """`Update` / Haiku scope ``light_global_update``."""

    def __init__(self, config, local_size: int, atom_count: int,
                 global_update: bool = True):
        super().__init__()
        c = self.config = config
        self.global_update = global_update
        self.linear_condition = HkLinear(local_size, local_size,
                                         bias=False, initializer="zeros")
        self.mlp = MLP(atom_count * 3, local_size * 2, out_size=local_size,
                       depth=2, activation=gelu, final_init="zeros")
        f = local_size * c.factor
        self.linear_update = HkLinear(local_size, f, bias=False, initializer="linear")
        self.linear_gate = HkLinear(local_size, f, bias=False, initializer="relu")
        if global_update:
            self.linear_chain_gate = HkLinear(local_size, f, bias=False, initializer="relu")
            self.linear_batch_gate = HkLinear(local_size, f, bias=False, initializer="relu")
        self.linear_out = HkLinear(f, local_size, initializer="zeros")

    def forward(self, local: Tensor, pos: Tensor, chain: Tensor, batch: Tensor,
                mask: Tensor, condition: Tensor | None = None) -> Tensor:
        _, _, local_pos = aa_frames_t(pos)
        local_pos = local_pos.reshape(local_pos.shape[0], -1)
        if condition is not None:
            local = local + self.linear_condition(condition)
        local = local + self.mlp(local_pos)

        local_update = self.linear_update(local)
        local_gate = gelu(self.linear_gate(local))
        hidden = local_gate * local_update
        if self.global_update:
            chain_gate = gelu(self.linear_chain_gate(local))
            batch_gate = gelu(self.linear_batch_gate(local))
            m = mask[..., None]
            hidden = hidden + index_mean(batch_gate * local_update, batch, m)
            hidden = hidden + index_mean(chain_gate * local_update, chain, m)
        return self.linear_out(hidden)


class EncoderUpdate(nn.Module):
    """`EncoderUpdate`: no conditioning, no global averaging. Same Haiku scope
    name (``light_global_update``) as `Update`, but only three linears."""

    def __init__(self, config, local_size: int, atom_count: int):
        super().__init__()
        c = self.config = config
        self.mlp = MLP(atom_count * 3, local_size * 2, out_size=local_size,
                       depth=2, activation=gelu, final_init="zeros")
        f = local_size * c.factor
        self.linear_update = HkLinear(local_size, f, bias=False, initializer="linear")
        self.linear_gate = HkLinear(local_size, f, bias=False, initializer="relu")
        self.linear_out = HkLinear(f, local_size, initializer="zeros")

    def forward(self, local: Tensor, pos: Tensor, chain: Tensor, batch: Tensor,
                mask: Tensor) -> Tensor:
        _, _, local_pos = aa_frames_t(pos)
        local_pos = local_pos.reshape(local_pos.shape[0], -1)
        local = local + self.mlp(local_pos)
        hidden = gelu(self.linear_gate(local)) * self.linear_update(local)
        return self.linear_out(hidden)


class UpdatePositions(nn.Module):
    """Frame-local additive position update (zero-init, so blocks start as no-ops)."""

    def __init__(self, local_size: int, atom_count: int):
        super().__init__()
        self.atom_count = atom_count
        self.linear = HkLinear(local_size, atom_count * 3, bias=False,
                               initializer="zeros")

    def forward(self, pos: Tensor, local_norm: Tensor, scale) -> Tensor:
        rot, trans, local_pos = aa_frames_t(pos)
        pos_update = self.linear(local_norm)
        if torch.is_tensor(scale):
            pos_update = scale * pos_update
        else:
            pos_update = float(scale) * pos_update
        pos_update = pos_update.reshape(*pos_update.shape[:-1], self.atom_count, 3)
        local_pos = local_pos + pos_update
        return apply_t(rot, trans, local_pos, k=1)


class DiffusionBlock(nn.Module):
    """One trunk layer: two pair+IPA branches, a global update, a position update."""

    def __init__(self, config, local_size: int, atom_count: int,
                 use_init_pos: bool = False):
        super().__init__()
        c = self.config = config
        p = c.pair_size
        self.pair_1 = DiffusionPairFeatures(c, atom_count, use_init_pos)
        self.layer_norm_1 = HkLayerNorm(local_size)
        self.attn_1 = SparseStructureAttention(c, local_size, p)
        self.pair_2 = DiffusionPairFeatures(c, atom_count, use_init_pos)
        self.layer_norm_3 = HkLayerNorm(local_size)
        self.attn_2 = SparseStructureAttention(c, local_size, p)
        self.layer_norm_4 = HkLayerNorm(local_size)
        self.update = Update(c, local_size, atom_count, global_update=True)
        self.layer_norm_5 = HkLayerNorm(local_size)
        self.update_positions = UpdatePositions(local_size, atom_count)
        # HOIST every config value `forward` needs into plain Python
        # attributes. `DotDict.__getattr__` is `dict.get`, which Dynamo cannot
        # trace: `_getattr_static` raises AttributeError and Dynamo BREAKS THE
        # GRAPH at every `config.<key>` read inside the compiled region.
        # Measured on `forward_core`: 27 graphs / 26 breaks before this, and
        # a broken graph is a lost fusion, i.e. extra kernels - which is the
        # one currency this model has. Reading them here, at construction
        # time, is free and bit-exact.
        self.sigma_data = float(c.sigma_data)
        self.precond_edm = c.preconditioning in ("edm", "edm_scaled")

    def forward(self, features: Tensor, pos: Tensor, prev_pos: Tensor,
                condition: Tensor, pair_condition: dict, time: Tensor,
                resi: Tensor, chain: Tensor, batch: Tensor, mask: Tensor,
                current_neighbours: Tensor, cond_neighbours: Tensor,
                cyclic_mask: Tensor | None = None,
                init_pos: Tensor | None = None) -> tuple[Tensor, Tensor]:
        sigma = self.sigma_data

        pair, pair_mask = self.pair_1(pos, pair_condition, current_neighbours,
                                      resi, chain, batch, mask,
                                      cyclic_mask=cyclic_mask, init_pos=init_pos)
        features = features + self.attn_1(
            self.layer_norm_1(features), pos / sigma, pair, pair_mask, current_neighbours)

        pair, pair_mask = self.pair_2(prev_pos, pair_condition, cond_neighbours,
                                      resi, chain, batch, mask,
                                      cyclic_mask=cyclic_mask, init_pos=init_pos)
        features = features + self.attn_2(
            self.layer_norm_3(features), prev_pos / sigma, pair, pair_mask, cond_neighbours)

        features = features + self.update(
            self.layer_norm_4(features), pos, chain, batch, mask, condition)

        local_norm = self.layer_norm_5(features)
        # `preconditioning_scale_factors(config, ...)["out"]` inlined: it reads
        # two config keys, and a config read is a graph break (see __init__).
        # Every non-EDM branch of it returns `sigma_data` unchanged.
        out_scale = (edm_scaling(sigma, 1.0, alpha=1.0, beta=time[:, None])["out"]
                     if self.precond_edm else sigma)
        pos = self.update_positions(pos, local_norm, out_scale)
        return features, pos.to(local_norm.dtype)


class DiffusionStack(nn.Module):
    """`diffusion_depth` independent blocks.

    Upstream runs these through `hk.LayerStack`, which stacks each layer's
    weights along a leading axis; here they are separate modules and the
    converter slices ``[i]``.
    """

    def __init__(self, config, local_size: int, atom_count: int,
                 use_init_pos: bool = False):
        super().__init__()
        self.config = config
        block_cls = MotifAdapterDiffusionBlock if config.multi_motif else DiffusionBlock
        self.blocks = nn.ModuleList([
            block_cls(config, local_size, atom_count, use_init_pos)
            for _ in range(config.diffusion_depth)
        ])

    def forward(self, local, pos, prev_pos, condition, pair_condition, time,
                resi, chain, batch, mask, neighbour_fn,
                cyclic_mask=None, init_pos=None):
        trajectory = []
        for block_idx, block in enumerate(self.blocks):
            current_neighbours, cond_neighbours = neighbour_fn(pos, prev_pos, block_idx)
            local, pos = block(local, pos, prev_pos, condition, pair_condition,
                               time, resi, chain, batch, mask,
                               current_neighbours, cond_neighbours,
                               cyclic_mask=cyclic_mask, init_pos=init_pos)
            trajectory.append(pos)
        return local, pos, torch.stack(trajectory, dim=0)


def extract_motif_neighbours(count: int):
    """Neighbours taken straight from the motif distance map.

    Unlike every other selector this ignores geometry entirely - it ranks by the
    supplied ``dmap`` and keeps only pairs the ``dmap_mask`` marks as known, so
    residues attend to their motif partners regardless of where the current
    structure has put them.
    """

    def inner(pair_condition: dict) -> Tensor:
        return get_neighbours(count)(pair_condition["dmap"],
                                     pair_condition["dmap_mask"], None)

    return inner


class MotifAdapterDiffusionBlock(nn.Module):
    """Trunk block for multi-motif scaffolding (`multimotif_vp`).

    Differs from `DiffusionBlock` in three ways, all of which matter:

    * the FIRST attention runs over motif neighbours, not sequence/spatial ones
    * that first attention takes its frames from ``prev_pos`` while its pair
      features come from ``pos`` - an asymmetry unique to this block
    * both pair blocks are given ``init_pos=prev_pos`` unconditionally, so this
      config always carries the two extra init_pos linears per pair block even
      though its preconditioning is plain VP
    """

    def __init__(self, config, local_size: int, atom_count: int,
                 use_init_pos: bool = True):
        super().__init__()
        c = self.config = config
        p = c.pair_size
        # always True here; the argument exists only to match DiffusionBlock
        self.pair_1 = DiffusionPairFeatures(c, atom_count, use_init_pos=True)
        self.layer_norm_1 = HkLayerNorm(local_size)
        self.attn_1 = SparseStructureAttention(c, local_size, p)
        self.pair_2 = DiffusionPairFeatures(c, atom_count, use_init_pos=True)
        self.layer_norm_3 = HkLayerNorm(local_size)
        self.attn_2 = SparseStructureAttention(c, local_size, p)
        self.layer_norm_4 = HkLayerNorm(local_size)
        self.update = Update(c, local_size, atom_count, global_update=True)
        self.layer_norm_5 = HkLayerNorm(local_size)
        self.update_positions = UpdatePositions(local_size, atom_count)
        # HOIST every config value `forward` needs into plain Python
        # attributes. `DotDict.__getattr__` is `dict.get`, which Dynamo cannot
        # trace: `_getattr_static` raises AttributeError and Dynamo BREAKS THE
        # GRAPH at every `config.<key>` read inside the compiled region.
        # Measured on `forward_core`: 27 graphs / 26 breaks before this, and
        # a broken graph is a lost fusion, i.e. extra kernels - which is the
        # one currency this model has. Reading them here, at construction
        # time, is free and bit-exact.
        self.sigma_data = float(c.sigma_data)
        self.precond_edm = c.preconditioning in ("edm", "edm_scaled")

    def forward(self, features: Tensor, pos: Tensor, prev_pos: Tensor,
                condition: Tensor, pair_condition: dict, time: Tensor,
                resi: Tensor, chain: Tensor, batch: Tensor, mask: Tensor,
                current_neighbours: Tensor, cond_neighbours: Tensor,
                cyclic_mask: Tensor | None = None,
                init_pos: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # `cond_neighbours` carries the motif neighbours for this block; the
        # caller supplies them so the stack keeps one neighbour_fn contract.
        sigma = self.sigma_data
        motif_neighbours = cond_neighbours

        pair, pair_mask = self.pair_1(pos, pair_condition, motif_neighbours,
                                      resi, chain, batch, mask,
                                      cyclic_mask=cyclic_mask, init_pos=prev_pos)
        # note: pair features from `pos`, attention frames from `prev_pos`
        features = features + self.attn_1(
            self.layer_norm_1(features), prev_pos / sigma, pair, pair_mask,
            motif_neighbours)

        pair, pair_mask = self.pair_2(pos, pair_condition, current_neighbours,
                                      resi, chain, batch, mask,
                                      cyclic_mask=cyclic_mask, init_pos=prev_pos)
        features = features + self.attn_2(
            self.layer_norm_3(features), pos / sigma, pair, pair_mask,
            current_neighbours)

        features = features + self.update(
            self.layer_norm_4(features), pos, chain, batch, mask, condition)

        local_norm = self.layer_norm_5(features)
        # `preconditioning_scale_factors(config, ...)["out"]` inlined: it reads
        # two config keys, and a config read is a graph break (see __init__).
        # Every non-EDM branch of it returns `sigma_data` unchanged.
        out_scale = (edm_scaling(sigma, 1.0, alpha=1.0, beta=time[:, None])["out"]
                     if self.precond_edm else sigma)
        pos = self.update_positions(pos, local_norm, out_scale)
        return features, pos.to(local_norm.dtype)
