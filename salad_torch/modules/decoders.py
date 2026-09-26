"""The four output heads hanging off the diffusion trunk.

`AADecoder` (sequence), `DistogramDecoder` (pairwise distance distribution) and
`AngleHead` (sidechain torsions) all read the *same* ``local`` tensor that the
trunk produced. Upstream writes ``local += ...`` inside the AA decoder, which in
JAX rebinds a local name; in PyTorch it would mutate the shared tensor and
silently corrupt the other three heads. Every accumulation here is therefore
out-of-place - that is a correctness requirement, not a style choice.

Submodule *names* mirror Haiku's auto-generated scopes (``linear``,
``linear_1``, ``layer_norm``, ``mlp``, ...) so checkpoint conversion is a direct
name mapping. `aa_decoder_pair_features` is a plain function upstream, so its
layers land in the enclosing ``aa_decoder_block`` scope and share that scope's
``linear`` counter - hence ``linear`` .. ``linear_5`` all live on the block.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..geometry import features as GF
from ..geometry.frames import extract_aa_frames
from ..geometry.neighbours import get_index_neighbours, get_spatial_neighbours
from ..geometry.rigid import vec_norm
from ..geometry.sidechain import single_protein_sidechains
from .. import opt
from .attention import SparseStructureAttention
from .basic import MLP, HkLayerNorm, HkLinear, gelu
from .blocks import EncoderUpdate

__all__ = ["one_hot", "AADecoderPairFeatures", "AADecoderBlock", "AADecoderStack",
           "AADecoder", "DistogramDecoder", "AngleHead"]


def one_hot(x: Tensor, n: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """``jax.nn.one_hot`` semantics: out-of-range indices give an all-zero row.

    `torch.nn.functional.one_hot` raises instead, and salad relies on the JAX
    behaviour - e.g. the unconditional DSSP sentinel is ``3`` fed into a
    3-class one-hot, and `AADecoder` feeds the mask token ``20`` into a
    21-class one-hot (in range) while `Condition` feeds ``3`` into 3 classes
    (out of range, so zeros).
    """
    return (x[..., None] == torch.arange(n, device=x.device)).to(dtype)


# --------------------------------------------------------------------------
# sequence decoder
# --------------------------------------------------------------------------
class AADecoderPairFeatures(nn.Module):
    """`aa_decoder_pair_features`.

    Same geometric features as `DiffusionPairFeatures`, minus the motif/pair
    conditioning, plus the neighbour's current amino-acid embedding - that last
    term is what makes the decoder autoregressive-capable.
    """

    def __init__(self, config, atom_count: int):
        super().__init__()
        c = self.config = config
        p = c.pair_size

        # creation order fixes the Haiku linear_N / layer_norm_N indices
        self.linear_relpos = HkLinear(66, p, bias=False)                  # linear
        self.linear_dist = HkLinear(400, p, bias=False)                   # linear_1
        self.linear_dir = HkLinear(atom_count * 3, p, bias=False)         # linear_2
        self.linear_rot = HkLinear(9, p, bias=False)                      # linear_3
        self.linear_pairvec = HkLinear(12 * atom_count, p, bias=False)    # linear_4
        # the LayerNorm is constructed before the Linear feeding it, because
        # Python builds `hk.LayerNorm(...)` before evaluating its argument
        self.layer_norm_aa = HkLayerNorm(p)                               # layer_norm
        self.linear_aa = HkLinear(21, p, bias=False)                      # linear_5
        self.layer_norm = HkLayerNorm(p)                                  # layer_norm_1
        self.mlp = MLP(p, p * 2, out_size=p, depth=2, activation=gelu,
                       final_init="linear")

    def forward(self, pos: Tensor, aa: Tensor, neighbours: Tensor, resi: Tensor,
                chain: Tensor, batch: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """`pos` is raw angstroms (not ``pos / sigma_data``)."""
        # -1 means "no neighbour"; the gathered row wraps to the last residue and
        # is garbage, neutralised here and used only to mask attention logits.
        # The pair *features* themselves are deliberately left unmasked.
        pair_mask = mask[:, None] & mask[neighbours] & (neighbours != -1)

        # no cyclic offsets in the AA decoder, even when c.cyclic is set
        if opt.GATHER_ONEHOT:   # P3, bit-exact
            pair = self.linear_relpos.forward_indexed(
                GF.sequence_relative_position(32, one_hot=False)(
                    resi, chain, batch, neighbours).long())
        else:
            relpos = GF.sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours)
            pair = self.linear_relpos(relpos)
        pair = pair + self.linear_dist(GF.distance_features(pos, neighbours, 0.0, 22.0))
        pair = pair + self.linear_dir(GF.direction_features(pos, neighbours))
        pair = pair + self.linear_rot(GF.position_rotation_features(pos, neighbours))
        pair = pair + self.linear_pairvec(GF.pair_vector_features(pos, neighbours))

        # gather happens *after* the LayerNorm, so masked-out neighbours pick up
        # a normalised (not raw) embedding - matching upstream
        # `aa` is an amino-acid index in [0, 20] (20 = MASK), so the gather is
        # in range by construction; P3 again.
        aa_embedding = self.layer_norm_aa(
            self.linear_aa.forward_indexed(aa) if opt.GATHER_ONEHOT
            else self.linear_aa(one_hot(aa, 21)))
        pair = pair + aa_embedding[neighbours]

        pair = self.layer_norm(pair)
        pair = self.mlp(pair)
        return pair, pair_mask


class AADecoderBlock(nn.Module):
    """One `aa_decoder_block`: pair features, sparse IPA, a local update."""

    def __init__(self, config, local_size: int, atom_count: int):
        super().__init__()
        c = self.config = config
        self.pair = AADecoderPairFeatures(c, atom_count)
        self.layer_norm_2 = HkLayerNorm(local_size)
        self.attn = SparseStructureAttention(c, local_size, c.pair_size,
                                             normalize=True)
        self.layer_norm_3 = HkLayerNorm(local_size)
        self.update = EncoderUpdate(c, local_size, atom_count)

    def forward(self, aa: Tensor, features: Tensor, pos: Tensor, neighbours: Tensor,
                resi: Tensor, chain: Tensor, batch: Tensor, mask: Tensor) -> Tensor:
        pair, pair_mask = self.pair(pos, aa, neighbours, resi, chain, batch, mask)
        # unlike DiffusionBlock, positions go into attention unscaled
        features = features + self.attn(
            self.layer_norm_2(features), pos, pair, pair_mask, neighbours)
        features = features + self.update(
            self.layer_norm_3(features), pos, chain, batch, mask)
        return features


class AADecoderStack(nn.Module):
    """`aa_decoder_stack`: `aa_decoder_depth` blocks over *shared* neighbours.

    Upstream runs these through `hk.LayerStack`, which stacks each layer's
    weights along a leading axis; here they are separate modules and the
    converter slices ``[i]``. Unlike `DiffusionStack` the neighbour set is
    computed once, in `AADecoder`, because positions never change in here.
    """

    def __init__(self, config, local_size: int, atom_count: int, depth: int):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            AADecoderBlock(config, local_size, atom_count) for _ in range(depth)
        ])
        self.layer_norm = HkLayerNorm(local_size)

    def forward(self, aa: Tensor, local: Tensor, pos: Tensor, neighbours: Tensor,
                resi: Tensor, chain: Tensor, batch: Tensor, mask: Tensor) -> Tensor:
        for block in self.blocks:
            local = block(aa, local, pos, neighbours, resi, chain, batch, mask)
        return self.layer_norm(local)


class AADecoder(nn.Module):
    """`aa_decoder` with ``aa_decoder_kind == "adm"``: a masked sequence model.

    At inference `Diffusion` calls `train_forward` with an all-mask sequence, so
    this runs as a single-shot 20-way classifier over the denoised structure.
    """

    def __init__(self, config, local_size: int, atom_count: int):
        super().__init__()
        c = self.config = config
        self.linear_aa_embed = HkLinear(21, local_size, initializer="glorot")  # linear
        self.aa_decoder_stack = AADecoderStack(c, local_size, atom_count,
                                               c.aa_decoder_depth)
        # two LayerNorms run back to back here (aa_decoder_stack/layer_norm then
        # aa_decoder/layer_norm); upstream really does that, so we keep both
        self.layer_norm = HkLayerNorm(local_size)
        self.linear_out = HkLinear(local_size, 20, bias=False,
                                   initializer="zeros")                   # linear_1

    # ------------------------------------------------------------------
    def forward(self, aa: Tensor, local: Tensor, pos: Tensor, resi: Tensor,
                chain: Tensor, batch: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(raw 20-way logits, decoder features)``."""
        # never `+=`, see docstring; P3 gather, `aa` is in [0, 20]
        local = local + (self.linear_aa_embed.forward_indexed(aa) if opt.GATHER_ONEHOT
                         else self.linear_aa_embed(one_hot(aa, 21)))

        # extract_neighbours(num_index=0, num_spatial=32, num_random=0), with the
        # two zero-count stages elided: get_index_neighbours(0) returns an (L, 0)
        # seed and get_random_neighbours(0) would build an (L, L) uniform only to
        # concatenate nothing. Identical output, no wasted allocation or RNG draw.
        neighbours = get_spatial_neighbours(32)(pos[:, 1], batch, mask)

        local = self.aa_decoder_stack(aa, local, pos, neighbours, resi, chain,
                                      batch, mask)
        local = self.layer_norm(local)
        return self.linear_out(local), local

    # ------------------------------------------------------------------
    def train_forward(self, aa: Tensor, local: Tensor, pos: Tensor, resi: Tensor,
                      chain: Tensor, batch: Tensor, mask: Tensor,
                      generator: torch.Generator | None = None
                      ) -> tuple[Tensor, Tensor, Tensor]:
        """Mask a random fraction of the sequence, then decode.

        This is also the *inference* path - `Diffusion` calls it with
        ``aa_gt`` already all-mask, which makes the corruption a no-op. That
        case is short-circuited so sampling consumes no randomness.

        Returns ``(log-softmax logits, decoder features, corrupt_aa)``.
        """
        if bool((aa == 20).all()):
            corrupt_aa = torch.zeros_like(aa, dtype=torch.bool)
        else:
            # one masking rate per batch element, broadcast to its residues
            rate = torch.rand(batch.shape, device=aa.device, dtype=torch.float32,
                              generator=generator)[batch]
            corrupt_aa = torch.rand(rate.shape, device=aa.device, dtype=rate.dtype,
                                    generator=generator) < rate
            aa = torch.where(corrupt_aa, torch.full_like(aa, 20), aa)
        logits, features = self(aa, local, pos, resi, chain, batch, mask)
        return torch.log_softmax(logits, dim=-1), features, corrupt_aa

    # ------------------------------------------------------------------
    def argmax(self, aa: Tensor, local: Tensor, pos: Tensor, resi: Tensor,
               chain: Tensor, batch: Tensor, mask: Tensor) -> Tensor:
        """Single-shot greedy decode from a fully masked sequence."""
        logits, _ = self(torch.full_like(aa, 20), local, pos, resi, chain,
                         batch, mask)
        return logits.argmax(dim=-1)

    def sample(self, aa: Tensor, local: Tensor, pos: Tensor, resi: Tensor,
              chain: Tensor, batch: Tensor, mask: Tensor, temperature: float,
              generator: torch.Generator | None = None) -> Tensor:
        """``config.sample_aa``'s categorical, random-order autoregressive decode.

        Unreachable for `default_vp` (`config.sample_aa` is unset there); this
        is what `--sample_final` (`notes/specs/06_generation_flags.md` §5.3)
        needs. Exactly `aa.shape[0]` iterations, one residue revealed per
        iteration, **re-running the whole decoder stack from scratch every
        time** - not an oversight to optimise away, upstream genuinely does
        this (there is no cheaper faithful alternative).

        Every iteration draws a full `(L, 20)` categorical sample even though
        only the newly-revealed residue's draw is kept - ported literally
        (not "optimised" to draw a single sample) so a fixed generator
        consumes the same amount of randomness upstream's JAX version does in
        spirit, even though the two PRNGs can never agree bit-for-bit.
        """
        L = aa.shape[0]
        aa = torch.full_like(aa, 20)
        temp = float(temperature) if temperature else 0.1
        for _ in range(L):
            logits, _ = self(aa, local, pos, resi, chain, batch, mask)
            logits = logits / temp
            u = torch.rand(L, generator=generator, device=aa.device,
                           dtype=local.dtype)
            u = torch.where(aa == 20, u, torch.full_like(u, float("inf")))
            index = int(torch.argmin(u))
            sampled = torch.multinomial(torch.softmax(logits, dim=-1), 1,
                                        generator=generator).squeeze(-1)
            aa = aa.clone()
            aa[index] = sampled[index]
        return aa


# --------------------------------------------------------------------------
# distogram decoder
# --------------------------------------------------------------------------
class DistogramDecoder(nn.Module):
    """`distogram_decoder`: a 64-bin distance distribution over 64 neighbours.

    Structurally an unstacked, single-shot `AADecoderPairFeatures` with the
    amino-acid term replaced by the two endpoints' ``local`` features.
    """

    def __init__(self, config, local_size: int, atom_count: int):
        super().__init__()
        c = self.config = config
        p = c.pair_size
        self.linear_relpos = HkLinear(66, p, bias=False)                  # linear
        self.linear_dist = HkLinear(400, p, bias=False)                   # linear_1
        self.linear_dir = HkLinear(atom_count * 3, p, bias=False)         # linear_2
        self.linear_rot = HkLinear(9, p, bias=False)                      # linear_3
        self.linear_pairvec = HkLinear(12 * atom_count, p, bias=False)    # linear_4
        self.linear_local_self = HkLinear(local_size, p, bias=False)      # linear_5
        self.linear_local_nb = HkLinear(local_size, p, bias=False)        # linear_6
        self.layer_norm = HkLayerNorm(p)
        self.mlp = MLP(p, p * 2, out_size=p, depth=2, activation=gelu,
                       final_init="zeros")

    def forward(self, local: Tensor, pos: Tensor, neighbours: Tensor, resi: Tensor,
                chain: Tensor, batch: Tensor, mask: Tensor,
                cyclic_mask: Tensor | None = None) -> Tensor:
        """Returns ``(L, K, 64)`` log-probabilities - already log-softmaxed."""
        if opt.GATHER_ONEHOT:   # P3, bit-exact
            pair = self.linear_relpos.forward_indexed(
                GF.sequence_relative_position(
                    32, one_hot=False, cyclic=cyclic_mask is not None)(
                        resi, chain, batch, neighbours,
                        cyclic_mask=cyclic_mask).long())
        else:
            relpos = GF.sequence_relative_position(
                32, one_hot=True, cyclic=cyclic_mask is not None)(
                    resi, chain, batch, neighbours, cyclic_mask=cyclic_mask)
            pair = self.linear_relpos(relpos)
        # guarded by `if not c.distogram_no_pos:` upstream, which is on for
        # every shipped config
        pair = pair + self.linear_dist(GF.distance_features(pos, neighbours, 0.0, 22.0))
        pair = pair + self.linear_dir(GF.direction_features(pos, neighbours))
        pair = pair + self.linear_rot(GF.position_rotation_features(pos, neighbours))
        pair = pair + self.linear_pairvec(GF.pair_vector_features(pos, neighbours))
        pair = pair + self.linear_local_self(local)[:, None]
        pair = pair + self.linear_local_nb(local)[neighbours]
        pair = self.layer_norm(pair)
        # upstream computes a pair_mask here and then never uses it
        return torch.log_softmax(self.mlp(pair), dim=-1)


# --------------------------------------------------------------------------
# sidechain torsion head
# --------------------------------------------------------------------------
class AngleHead(nn.Module):
    """`get_angle_positions`: 7 torsions per residue, then atom14 coordinates.

    The MLP is created directly in the ``diffusion`` scope upstream, so its
    checkpoint path is ``diffusion/mlp`` (``diffusion/linear`` is a different
    layer - the DSSP decoder).
    """

    def __init__(self, local_size: int, atom_count: int):
        super().__init__()
        self.atom_count = atom_count
        # local + frame-local coords + their distance RBF + one-hot aa
        in_size = local_size + atom_count * 3 + atom_count * 16 + 21
        self.mlp = MLP(in_size, 256, out_size=14, depth=2, activation=gelu,
                       bias=False, init="relu", final_init="linear")

    def forward(self, aatype: Tensor, local: Tensor,
                pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(raw_angles (L,7,2), angles (L,7,2), atom14 pos (L,14,3))``.

        ``angles[..., 0]`` is sin and ``angles[..., 1]`` is cos.
        """
        frames, local_positions = extract_aa_frames(pos)
        n = local.shape[0]
        features = torch.cat((
            local,
            local_positions.reshape(n, -1),
            GF.distance_rbf(vec_norm(local_positions), 0.0, 10.0, 16).reshape(n, -1),
            one_hot(aatype, 21),
        ), dim=-1)
        raw_angles = self.mlp(features).reshape(n, 7, 2)
        # normalise to the unit circle; the epsilon floors the *squared* sum
        angles = raw_angles / torch.sqrt(torch.clamp(
            (raw_angles ** 2).sum(dim=-1, keepdim=True), min=1e-6))
        angle_pos, _ = single_protein_sidechains(aatype, frames, angles)
        # N, CA, C, O come straight from the trunk; only CB onwards is predicted
        angle_pos = torch.cat((pos[..., :4, :], angle_pos[..., 4:, :]), dim=-2)
        return raw_angles, angles, angle_pos
