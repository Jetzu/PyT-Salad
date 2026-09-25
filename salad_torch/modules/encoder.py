"""The salad `Encoder`: invents 15 learned pseudo-atom positions per residue.

Called exactly once per training example, on the CLEAN (t=0) structure, inside
`StructureDiffusion.encode` -> `StructureDiffusion.prepare_data` (spec-05 §0-1).
It is never called at de-novo sampling time (there is no atom14 ground truth to
encode then). Its output becomes `data["pos"]`, the ``(L, 20, 3)`` tensor the
diffusion trunk is trained to denoise - so despite being "just" a preprocessing
step, it has real trainable weights that must be jointly optimised, and getting
it wrong silently corrupts every position/rotation/FAPE/distogram loss.

Two upstream facts drive everything here:

  (a) The Encoder is **sequence-agnostic** - `aa`/`aa_gt` never enters
      `prepare_features` or `EncoderBlock` (contrast the dead `Atom14Encoder`,
      spec-05 §7, which *is* aa-conditioned; not implemented here - unreachable
      for every released config, see `config.encode_atom14`).
  (b) **Only atoms 5:20** (the 15 "augment" pseudo-atoms) are ever produced or
      changed. Atoms 0:5 (N, CA, C, O, idealised pseudo-CB) pass through this
      module completely unchanged, both at input and output.

Caller contract (not this module's problem, but easy to get wrong): the
`StructureDiffusion.encode` call site writes ``data["pos"] = backbone5`` right
before invoking this module, then later returns a *freshly built* dict whose
``"pos"`` is the 20-atom output - that intermediate 5-atom write is stale and
never read again upstream. A port that mutates a shared `data` dict in place
(rather than building its return value from local variables) must not let that
stale 5-atom write leak out as a final ``"pos"``.

Haiku scope: ``structure_diffusion/~encode/encoder/...`` (see
`salad_torch/convert/haiku_to_torch.py`'s `encoder_map`/`encoder_block_map`/
`load_encoder` for the checkpoint-loading side of this).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from ..geometry import features as GF
from ..geometry.frames import extract_aa_frames
from ..geometry.neighbours import get_spatial_neighbours
from ..geometry.rigid import vec_norm, vec_normalized
from .attention import SparseStructureAttention
from .basic import MLP, HkLayerNorm, HkLinear, gelu, init_weight_
from .blocks import EncoderUpdate

__all__ = ["VectorLinear", "vector_mean_norm", "EncoderPairFeatures",
           "EncoderBlock", "EncoderStack", "Encoder"]


class VectorLinear(nn.Module):
    """Equivariant linear recombination of position vectors, along the ATOM axis.

    Haiku's `VectorLinear` (`geometric.py:282-301`) swaps the atom and xyz axes,
    runs a plain `Linear` (which contracts the *last* axis), then swaps back -
    the point is that the contraction happens over atoms, not xyz, so a fixed
    real-valued recombination of the input vectors stays SE(3)-equivariant
    (mixing xyz would not be). The single einsum below does the same
    contraction directly: ``out[..., o, c] = sum_a vec[..., a, c] * w[a, o]``.

    Subtlety (checklist #1): `Linear.__call__` upstream *always* allocates a
    bias, and `VectorLinear` hardcodes ``bias=False`` at its call site - so
    ``b`` sits in every checkpoint (shape ``(out_size,)``) but is never added
    to the output. It is allocated here too, for state-dict parity with the
    checkpoint, and simply never read in `forward`.

    Only one instance exists in the whole model (`Encoder.prepare_features`
    step 3): ``VectorLinear(5, 15)`` projecting the 5 fixed backbone/pseudo-CB
    local coordinates onto 15 augment-atom coordinates. It is explicitly
    constructed with ``initializer="linear"`` there, NOT this class's own
    default (`VectorLinear`'s Haiku default is "zeros" - the call site
    overrides it, so this is not a small perturbation at initialisation).
    """

    def __init__(self, in_size: int, out_size: int, initializer: str = "zeros"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.w = nn.Parameter(torch.empty(in_size, out_size))
        self.b = nn.Parameter(torch.zeros(out_size))  # allocated, never applied
        init_weight_(self.w, initializer)

    def forward(self, vec: Tensor) -> Tensor:
        """``vec``: ``(..., A_in, 3)`` -> ``(..., A_out, 3)``. Bias never applied."""
        return torch.einsum("...ac,ao->...oc", vec, self.w)


def vector_mean_norm(vec: Tensor, eps: float = 1e-6) -> Tensor:
    """Recentre a per-residue point cloud, then rescale by its MEAN atom-distance.

    `geometric.py:358-365`. This is **not** per-atom unit normalization:

      1. subtract the centroid over the ATOM axis (`vec` becomes mean-zero);
      2. compute each atom's distance from that new centre;
      3. average those per-atom distances over the atom axis (one scalar per
         residue);
      4. divide the centred cloud by that single scalar.

    Net effect: the output atoms keep their relative directions/spread but are
    recentred to mean-zero and globally rescaled so their *average* distance
    from the new centre is 1.

    Subtlety (checklist #2): the clamp here is ``max(sum_xyz(x**2), 1e-6)`` -
    ``eps`` directly on the SQUARED sum. This port's usual convention
    (`geometry.rigid.vec_norm`, ``_NORM_EPS=1e-6``) clamps at ``eps**2=1e-12``
    on the squared sum instead - a 1e6x tighter floor. Do not substitute
    `vec_norm`/`vec_normalized` here; upstream really does use the looser
    floor for this one function.

    Called twice in the whole `Encoder`: once on the freshly-projected augment
    atoms (`prepare_features`, over the 15 augment atoms only) and once on the
    refined augment atoms at the end of `Encoder.forward` (also 15 atoms only
    - this is never applied to all 20).
    """
    vec = vec - vec.mean(dim=-2, keepdim=True)
    per_atom_norm = torch.sqrt(torch.clamp((vec ** 2).sum(dim=-1, keepdim=True), min=eps))
    scale = per_atom_norm.mean(dim=-2, keepdim=True)
    return vec / scale


class EncoderPairFeatures(nn.Module):
    """`encoder_pair_features` (NSB:1845-1866): geometry-only pair features.

    Structurally `AADecoderPairFeatures` (`decoders.py`) minus the amino-acid
    embedding term - same feature functions, same order, same sizes. Like that
    function, `encoder_pair_features` is a plain function upstream (not an
    `hk.Module`), so its Linears/LayerNorm/MLP share the enclosing
    `encoder_block` scope's flat `linear`/`layer_norm` counters rather than
    getting a scope of their own - see `EncoderBlock`'s docstring and
    `haiku_to_torch.encoder_block_map` for the exact name assignment.
    """

    def __init__(self, config, atom_count: int = 20):
        super().__init__()
        c = self.config = config
        p = c.pair_size
        # creation order fixes the Haiku linear_N / layer_norm index (see
        # encoder_block_map): linear..linear_4, mlp, then this LayerNorm takes
        # the first bare `layer_norm` name in the enclosing block scope.
        self.linear_relpos = HkLinear(66, p, bias=False)
        self.linear_dist = HkLinear(400, p, bias=False)
        self.linear_dir = HkLinear(atom_count * 3, p, bias=False)
        self.linear_rot = HkLinear(9, p, bias=False)
        self.linear_pairvec = HkLinear(12 * atom_count, p, bias=False)
        self.layer_norm = HkLayerNorm(p)
        self.mlp = MLP(p, p * 2, out_size=p, depth=2, activation=gelu, final_init="linear")

    def forward(self, pos: Tensor, neighbours: Tensor, resi: Tensor, chain: Tensor,
                batch: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        mask = mask.bool()
        pair_mask = mask[:, None] & mask[neighbours] & (neighbours != -1)
        pair = self.linear_relpos(GF.sequence_relative_position(32, one_hot=True)(
            resi, chain, batch, neighbours))
        pair = pair + self.linear_dist(GF.distance_features(pos, neighbours, 0.0, 22.0))
        pair = pair + self.linear_dir(GF.direction_features(pos, neighbours))
        pair = pair + self.linear_rot(GF.position_rotation_features(pos, neighbours))
        pair = pair + self.linear_pairvec(GF.pair_vector_features(pos, neighbours))
        pair = self.layer_norm(pair)
        pair = self.mlp(pair)
        return pair, pair_mask


class EncoderBlock(nn.Module):
    """`EncoderBlock` (NSB:816-838): pair features + sparse IPA + a local update.

    `c.resi_dual` is hardcoded to the ``False`` (prenorm-skip) behaviour -
    ``local + out`` after a plain `LayerNorm(local)` input - matching every
    other block in this port (`DiffusionBlock`/`AADecoderBlock`). The
    `resi_dual == True` branch (ResiDual, tuple-valued features) is not
    implemented anywhere in this codebase; do not build a dispatch for it.

    Positions are passed to attention **unscaled** (no ``/sigma_data``) -
    matching `AADecoderBlock`'s convention, not `DiffusionBlock`'s. The Encoder
    has no notion of a diffusion noise scale at all.
    """

    def __init__(self, config, local_size: int = 128, atom_count: int = 20):
        super().__init__()
        c = self.config = config
        self.pair_features = EncoderPairFeatures(c, atom_count)
        # two independent LayerNorm(128) instances, per Haiku creation order:
        # layer_norm_1 feeds attention, layer_norm_2 feeds the update.
        self.layer_norm_1 = HkLayerNorm(local_size)
        self.attn = SparseStructureAttention(c, local_size=local_size,
                                             pair_size=c.pair_size, normalize=True)
        self.layer_norm_2 = HkLayerNorm(local_size)
        # EncoderUpdate == upstream's `Update(c, global_update=False)` called
        # with condition=None; already implemented verbatim in blocks.py.
        self.update = EncoderUpdate(c, local_size=local_size, atom_count=atom_count)

    def forward(self, features: Tensor, pos: Tensor, neighbours: Tensor, resi: Tensor,
                chain: Tensor, batch: Tensor, mask: Tensor) -> Tensor:
        pair, pair_mask = self.pair_features(pos, neighbours, resi, chain, batch, mask)
        features = features + self.attn(
            self.layer_norm_1(features), pos, pair, pair_mask, neighbours)
        features = features + self.update(
            self.layer_norm_2(features), pos, chain, batch, mask)
        return features


class EncoderStack(nn.Module):
    """`EncoderStack` (NSB:848-870), Haiku default_name ``self_cond_stack``.

    Subtlety (checklist #3): unlike `DiffusionStack`, ``pos``/``neighbours``
    are **fixed for the whole stack** - every block reads the exact same
    values computed once in `Encoder.prepare_features`. Only `local` threads
    through block to block. Feeding an updated `pos` into a later block would
    silently diverge from upstream.

    Upstream stacks `encoder_depth` independent-weight blocks via
    `hk.experimental.layer_stack` (traced once, weights carry a leading depth
    axis in the checkpoint); here they are separate `EncoderBlock` instances
    and the converter slices ``[i]`` off that axis, matching the precedent set
    by `DiffusionStack`/`AADecoderStack`. `hk.remat` wrapping each block
    upstream is a pure memory/compute tradeoff with no numerical effect and is
    not reproduced here, matching those same two stacks.
    """

    def __init__(self, config, local_size: int = 128, atom_count: int = 20):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            EncoderBlock(config, local_size, atom_count)
            for _ in range(config.encoder_depth)
        ])
        self.layer_norm = HkLayerNorm(local_size)  # single LayerNorm AFTER the whole stack

    def forward(self, local: Tensor, pos: Tensor, neighbours: Tensor, resi: Tensor,
                chain: Tensor, batch: Tensor, mask: Tensor) -> Tensor:
        for block in self.blocks:
            local = block(local, pos, neighbours, resi, chain, batch, mask)
        return self.layer_norm(local)


class Encoder(nn.Module):
    """`Encoder` (NSB:700-773): builds the ``(L, 20, 3)`` training-target positions.

    Config: `default_vp` (`config.py`) - ``augment_size=15``, ``encoder_depth=2``,
    ``local_size=128``, ``pair_size=64``, ``num_neighbours=48``. All four
    released configs share this exact Encoder untouched.
    """

    def __init__(self, config):
        super().__init__()
        c = self.config = config
        self.augment_size = c.augment_size            # 15
        # matches the existing expression at salad_torch/modules/diffusion.py:85
        # (checklist #10: 20, not the atom14 axis (14) or backbone-only (5))
        self.atom_count = 5 + c.augment_size
        local_size = c.local_size

        # --- prepare_features (Haiku auto-scope ~prepare_features) ---------
        # NOT VectorLinear's own "zeros" default - this call site explicitly
        # overrides it with a linear (VarianceScaling) init.
        self.vector_linear = VectorLinear(5, self.augment_size, initializer="linear")
        # bias=False on BOTH internal Linears here (checklist #1); `b` is still
        # allocated per HkLinear's convention, just never added.
        self.local_mlp = MLP(400, local_size * 4, out_size=local_size, depth=2,
                             activation=gelu, bias=False, final_init="linear")
        self.local_layer_norm = HkLayerNorm(local_size)

        self.encoder_stack = EncoderStack(c, local_size=local_size, atom_count=self.atom_count)

        # --- position-update MLP, directly in the `encoder` scope ----------
        # bias=True on BOTH internal Linears (checklist #1) - unlike local_mlp
        # above, this MLP really applies its bias.
        #
        # Subtlety (checklist #6): final_init="linear" here is upstream's
        # init_linear() (VarianceScaling(fan_in, scale=1.0)) - NOT zero-init,
        # unlike almost every other "final" delta projection in this port
        # (UpdatePositions.linear, Update.linear_out both use init_zeros()).
        # This is a deliberate upstream oddity: at initialisation the Encoder's
        # augment positions can move by an O(1) amount from the initial
        # VectorLinear projection, not a small perturbation of it. Do NOT
        # "fix" this to zero-init by analogy with the rest of the codebase.
        self.update_mlp = MLP(local_size, local_size * 2, out_size=self.augment_size * 3,
                              depth=2, activation=gelu, bias=True, final_init="linear")

    # ----------------------------------------------------------------------
    def prepare_features(self, data: dict):
        """`Encoder.prepare_features` (NSB:700-741): build the 20-atom input.

        Reads ``data["pos"]`` (``(L, 5, 3)``, the N/CA/C/O/pseudo-CB backbone),
        ``data["residue_index"]``, ``data["chain_index"]``,
        ``data["batch_index"]``, ``data["mask"]``.

        Returns ``(local, positions, neighbours, resi, chain, batch, mask)``:
        ``local`` is ``(L, 128)``, ``positions`` is ``(L, 20, 3)`` in the
        GLOBAL frame, ``neighbours`` is ``(L, 48)``.
        """
        positions = data["pos"]
        resi, chain = data["residue_index"], data["chain_index"]
        batch, mask = data["batch_index"], data["mask"]

        frames, local_positions = extract_aa_frames(positions)   # (L,) rigid; (L, 5, 3)

        augment = self.vector_linear(local_positions)              # (L, 15, 3)
        augment = vector_mean_norm(augment)

        local_positions = torch.cat([local_positions[:, :5], augment], dim=-2)  # (L, 20, 3)
        positions = frames[:, None].apply_to_point(local_positions)              # (L, 20, 3) GLOBAL

        # The Encoder's ONLY neighbour-selection stage: a single spatial-only
        # k=48 stage (unlike the diffusion trunk's 3-stage index+spatial+
        # random `extract_neighbours`), anchored on atom index 4 (the
        # pseudo-CB, NOT an augment atom).
        #
        # Subtlety (checklist #8 - cross-chain neighbours): `item=batch_index`,
        # not `chain` - a residue's 48 spatial neighbours may cross chain
        # boundaries within the same batch item. Deliberate: the Encoder needs
        # whole-complex context to place pseudo-atoms sensibly at interfaces.
        neighbours = get_spatial_neighbours(self.config.num_neighbours)(
            positions[:, 4], batch, mask)

        # Subtlety (checklist #9): upstream recomputes `extract_aa_frames` on
        # this GLOBAL `positions` here, discarding the recomputed frames (they
        # are mathematically identical to `frames` above - N/CA/C round-trip
        # unchanged through apply_to_point . apply_inverse_to_point) and
        # reusing only the local coordinates, which likewise reproduce
        # `local_positions` above up to fp32 rounding. Safe to elide the
        # recompute and reuse `local_positions` directly, as done here.
        dist = vec_norm(local_positions)                            # (L, 20)

        n = local_positions.shape[0]
        local_features = torch.cat([
            vec_normalized(local_positions).reshape(n, -1),                # (L, 60)
            GF.distance_rbf(dist, 0.0, 22.0, bins=16).reshape(n, -1),      # (L, 320)
            torch.log(dist + 1.0),                                          # (L, 20); log(dist+1), not log1p
        ], dim=-1)                                                          # (L, 400)

        local = self.local_mlp(local_features)
        local = self.local_layer_norm(local)

        return local, positions, neighbours, resi, chain, batch, mask

    # ----------------------------------------------------------------------
    def forward(self, data: dict) -> Tuple[Tensor, Tensor]:
        """`Encoder.__call__` (NSB:745-773).

        Returns ``(local, pos)``: ``local`` is ``(L, 128)``, the encoder's
        final per-residue feature vector - it carries real gradient (it feeds
        `update_mlp` below) but is discarded by the caller
        (`StructureDiffusion.encode` only forwards `pos`; checklist #5),
        so it still must be computed here, just not propagated further.
        ``pos`` is ``(L, 20, 3)`` - the real output, and the diffusion trunk's
        training target. Atoms 0:5 of ``pos`` are bit-for-bit (up to one
        fp32 frame-apply/apply-inverse round trip) the same as the 5-atom
        backbone passed in via ``data["pos"]``; only atoms 5:20 differ
        (checklist #4).
        """
        local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data)

        local = self.encoder_stack(local, pos, neighbours, resi, chain, batch, mask)

        # `pos` here is the ORIGINAL prepare_features output - the stack never
        # updates it (EncoderStack subtlety, see its docstring). Re-derive
        # frames/local coordinates fresh from it, exactly as upstream does;
        # `frames` is identical to the one computed inside prepare_features
        # since N/CA/C never changed.
        frames, local_positions = extract_aa_frames(pos)
        augment = local_positions[:, 5:]                # (L, 15, 3) ORIGINAL, pre-refinement

        update = self.update_mlp(local).reshape(-1, self.augment_size, 3)
        # ordinary (out-of-place) addition - JAX `+=` on an immutable array is
        # not an in-place mutation, and this port never mutates a possibly-
        # shared tensor in place either.
        augment = augment + update
        augment = vector_mean_norm(augment)              # re-normalize the REFINED cloud

        local_positions = torch.cat([local_positions[:, :5], augment], dim=-2)  # (L, 20, 3)
        pos = frames[:, None].apply_to_point(local_positions)                    # (L, 20, 3) GLOBAL

        return local, pos
