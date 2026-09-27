"""The `Diffusion` network: feature preparation, trunk, and output heads.

Mirrors `salad.modules.noise_schedule_benchmark.Diffusion`. Haiku scope
``structure_diffusion/diffusion``.

Feature widths for `default_vp` (A=20 atoms, local_size=128):
  time 1+256, frame-local coords A*3=60, distance RBF A*64=1280, prev.local 128
  -> 1725, matching the checkpoint's ``~prepare_features/mlp/linear`` (1725, 512).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..diffusion.precond import edm_scaling, preconditioning_scale_factors
from ..geometry import features as GF
from ..geometry.frames import aa_frames_t
from ..geometry.neighbours import (extract_neighbours, get_index_neighbours,
                                   get_random_neighbours, get_spatial_neighbours)
from ..geometry.rigid import vec_norm
from .basic import MLP, HkLayerNorm, HkLinear, gelu
from ..config import uses_init_pos
from .. import opt
from .blocks import (DiffusionStack, condition_neighbour_prologue,
                     condition_neighbour_tail, extract_condition_neighbours,
                     extract_motif_neighbours)
from .decoders import AADecoder, AngleHead, DistogramDecoder

__all__ = ["Diffusion", "uniform_draw_count"]

def _unbind_draws(stack, count: int) -> list:
    """Split a pre-drawn ``(draws, N, N)`` stack into a plain Python list.

    Replaces a former ``_slice(stack, i)`` helper, and the replacement is worth
    a paragraph. Dynamo compiled `_slice` as its OWN frame and specialised it on
    the integer `i`; with 19 draws that blows the default
    ``cache_size_limit`` of 8, and every call past the eighth **breaks the
    graph**. Measured on `forward_core` before the change:
    ``torch._dynamo.explain`` reported 27 graphs / 26 breaks. Unbinding once,
    up front, into a list indexed by an int known at trace time gives Dynamo
    nothing to specialise on.

    `torch.unbind` returns views, so this costs no kernels and no memory. A
    2-D ``(N, N)`` field means "use this one field for every draw", which is
    what the JAX parity fixtures record (their generator patches one field into
    every call). Indexing that as a stack would silently hand a single ROW to
    the Gumbel and quietly corrupt neighbour selection - it cost 300-1800x
    parity on default_vp_scaled and multimotif_vp while default_vp happened to
    survive it.
    """
    if stack is None:
        return [None] * count
    if stack.dim() == 2:
        return [stack] * count
    return list(torch.unbind(stack, dim=0))


def _slice(stack, i: int):
    """Pick draw `i` from a pre-drawn ``(draws, N, N)`` stack.

    CUDA graphs cannot capture an RNG op ("Offset increment outside graph
    capture"), so every Gumbel-top-k draw is hoisted out and handed in here.

    NB this is indexed by an explicit integer rather than a stateful counter.
    A mutable counter object is a Dynamo *guard*, so `neighbour_fn` recompiles
    once per counter value and promptly blows the recompile cache limit; an
    integer known at trace time is free.

    A 2-D ``(N, N)`` field means "use this one field for every draw", which is
    what the JAX parity fixtures record (their generator patches one field into
    every call). Indexing that as a stack would silently hand a single ROW to
    the Gumbel and quietly corrupt neighbour selection - it cost 300-1800x
    parity on default_vp_scaled and multimotif_vp while default_vp happened to
    survive it.
    """
    if stack is None:
        return None
    return stack if stack.dim() == 2 else stack[i]


def uniform_draw_count(config) -> int:
    """How many (N, N) uniform fields one forward consumes.

    1 for the distogram neighbours, then per block: 1 for extract_neighbours
    and, unless multi_motif routes to the motif map, 2 for
    extract_condition_neighbours (its random and block-contact stages).

    This stays at 3 fields per block even when P5 drops the block-contact
    stage. The stack LAYOUT is what maps a seed to a trajectory: renumbering
    it would hand the surviving Gumbel stage a different field and generate a
    different (equally valid, but different) protein, which is not a change
    the exact path is allowed to make. The unconsumed slice costs one
    `(N, N)` slab of an already-single `rand` call.
    """
    per_block = 1 if config.multi_motif else 3
    return 1 + config.diffusion_depth * per_block



def vp_time_embedding(t: Tensor, size: int = 256) -> Tensor:
    """``concat(t, fourier(t))`` -> ``(L, 1 + size)``."""
    return torch.cat((t[:, None], GF.fourier_time_embedding(t, size=size)), dim=-1)


def edm_time_embedding(t: Tensor, size: int = 256) -> Tensor:
    """``concat(log(t)/4, fourier(log(t)/4))`` -> ``(L, 1 + size)``."""
    lt = torch.log(t) / 4
    return torch.cat((lt[:, None], GF.fourier_time_embedding(lt, size=size)), dim=-1)


class Diffusion(nn.Module):
    """Haiku scope ``diffusion``."""

    # time embeddings: VP-family kinds use the linear form, everything else log
    _VP_KINDS = ("vp", "vpfixed", "chroma", "flow")

    def __init__(self, config):
        super().__init__()
        c = self.config = config
        self.atom_count = 14 if c.encode_atom14 else 5 + c.augment_size
        a = self.atom_count
        ls = c.local_size
        # two independent routes: EDMDiffusionStack supplies init_pos for
        # edm_scaled, and the motif block passes init_pos=prev_pos always
        self.use_init_pos = uses_init_pos(c)

        # -- prepare_features (Haiku scope `~prepare_features`) --------------
        time_width = (1 + 256) if c.time_embedding else 0
        feat_width = time_width + a * 3 + a * 64 + ls
        if c.diffuse_sequence:
            feat_width += (1 + 256) + 21
        self.prepare_mlp = MLP(feat_width, 4 * ls, out_size=ls, depth=2,
                               activation=gelu, bias=False, final_init="linear")
        self.prepare_layer_norm_prev = HkLayerNorm(ls)   # layer_norm   -> prev.local
        self.prepare_layer_norm_local = HkLayerNorm(ls)  # layer_norm_1 -> MLP output
        self.prepare_layer_norm_cond = HkLayerNorm(ls)   # layer_norm_2 -> condition

        # -- trunk -----------------------------------------------------------
        self.diffusion_stack = DiffusionStack(c, ls, a, use_init_pos=self.use_init_pos)

        # -- heads -------------------------------------------------------------
        self.aa_decoder = AADecoder(c, ls, a)
        self.dssp_decoder = HkLinear(ls, 3, bias=False, initializer="zeros")
        self.distogram_decoder = DistogramDecoder(c, ls, a)
        self.angle_head = AngleHead(ls, a)

        # -- config, hoisted into plain attributes ---------------------------
        # `DotDict.__getattr__` is `dict.get`; Dynamo's `_getattr_static`
        # raises on it and BREAKS THE GRAPH at every `config.<key>` read inside
        # `forward_core`. Measured before this hoist: 30 graphs / 29 breaks,
        # every one of them a lost fusion. Reading the keys once, here, is
        # free and bit-exact. Any new config read in a hot path belongs here,
        # not in `forward`.
        self.multi_motif = bool(c.multi_motif)
        self.time_embedding_on = bool(c.time_embedding)
        self.diffuse_sequence = bool(c.diffuse_sequence)
        self.fape_neighbours = int(c.fape_neighbours)
        self.preconditioning = str(c.preconditioning or "")
        self.sigma_data = float(c.sigma_data)
        self.vp_time_embedding = c.diffusion_kind in self._VP_KINDS
        self.n_draws = uniform_draw_count(c)
        self.cyclic = bool(c.cyclic)
        self.depth = int(c.diffusion_depth)

    # ------------------------------------------------------------------
    def time_embedding(self, t: Tensor) -> Tensor:
        # `self.vp_time_embedding`, not `self.config.diffusion_kind` - see
        # __init__: a DotDict read inside the compiled region breaks the graph.
        if self.vp_time_embedding:
            return vp_time_embedding(t)
        return edm_time_embedding(t)

    def prepare_features(self, data: dict, prev: dict) -> tuple:
        c = self.config
        pos = data["pos_noised"]
        t_pos, t_seq = data["t_pos"], data["t_seq"]
        seq = data["seq_noised"]
        n = pos.shape[0]

        _, _, local_pos = aa_frames_t(pos)
        parts = [
            local_pos.reshape(n, -1),
            GF.distance_rbf(vec_norm(local_pos), 0.0, 22.0).reshape(n, -1),
        ]
        if self.time_embedding_on:
            parts = [self.time_embedding(t_pos)] + parts
        if self.diffuse_sequence:
            parts += [self.time_embedding(t_seq),
                      (seq[..., None] == torch.arange(21, device=seq.device)).float()]
        parts.append(self.prepare_layer_norm_prev(prev["local"]))

        local = self.prepare_mlp(torch.cat(parts, dim=-1))
        local = self.prepare_layer_norm_local(local)
        condition = self.prepare_layer_norm_cond(data["condition"])
        return (local, pos, prev["pos"], condition, data["pair_condition"],
                t_pos, t_seq, data["residue_index"], data["chain_index"],
                data["batch_index"], data["mask"])


    # ------------------------------------------------------------------
    # Tensor-in / tensor-out core.
    #
    # This exists purely for speed. The model is launch-bound - at L=100 one
    # DiffusionBlock costs ~8 ms on an H100 against JAX's ~3.1 ms for the whole
    # forward - so the win is not arithmetic, it is eliminating ~1500 kernel
    # launches per step. That needs `torch.compile`, ideally with CUDA graphs,
    # and Dynamo will not trace the python-dict shuffling the dict-based
    # forward does (it silently drops keys). Keeping a flat tensor signature
    # here makes the whole denoiser capturable in one graph.
    # ------------------------------------------------------------------
    def forward_core(self, pos_noised: Tensor, prev_pos: Tensor, prev_local: Tensor,
                     condition: Tensor, dmap: Tensor, omap: Tensor,
                     dmap_mask: Tensor, flags: Tensor,
                     t_pos: Tensor, t_seq: Tensor, seq_noised: Tensor,
                     aa_gt: Tensor, pos_gt: Tensor,
                     resi: Tensor, chain: Tensor, batch: Tensor, mask: Tensor,
                     cyclic_mask: Tensor | None = None,
                     uniform: Tensor | None = None,
                     need_aux: bool = True,
                     need_aa: bool = True,
                     pair_cond: bool = True,
                     ) -> tuple[Tensor, ...]:
        """Returns (pos, local, trajectory, aa_logits, aa_features, dssp,
        distogram_neighbours, raw_angles, angles, angle_pos, aatype), or just
        ``(pos, local, trajectory)`` when `need_aa` is False.

        `need_aux=False` skips the two outputs the sampler never reads: the
        dssp head and the distogram neighbour draw. The latter is a full
        (N, N) Gumbel + stable argsort, i.e. one of the more expensive single
        ops in the forward, spent on a tensor that is then discarded.

        `need_aa=False` (P2) additionally skips the whole AA decoder, the dssp
        head and the angle head. Legal only on a sampling step whose outputs
        are never read: the trunk's inputs are
        ``(pos_noised, prev_pos, prev_local, condition, aa_gt)`` and `aa_gt` is
        a base key that never changes over a run, so nothing the three heads
        produce can re-enter the model. `pos`, `local` and `trajectory` - the
        only things the sampler feeds back - are **bit-identical**, because
        this is a skip and not a reassociation.

        `pair_cond=False` (P4 + P5) asserts that this design has no pair
        conditioning at all: ``dmap_mask`` all false AND ``flags`` all false.
        The caller must establish that from the real tensors; getting it wrong
        is silently wrong, so `FastDenoiser` derives it and never guesses.
        Under that condition the `dmap`/`flags` pair terms are exactly-zero
        matrices fed through exactly-zero GEMMs (bit-exact to drop), and the
        fourth neighbour stage's every column is the `-1` sentinel, so the
        self-conditioning branch runs at K=48 instead of K=64 (reassociation
        of reductions over K whose dropped terms are exactly 0.0).

        These three flags are Python bools, so Dynamo specialises on them and
        each combination gets its own graph.
        """
        # NB every config value used below is a plain attribute hoisted in
        # __init__, never `self.config.<key>`: a DotDict read inside the
        # compiled region breaks the Dynamo graph (see __init__).
        # multi_motif ROUTES through the pair conditioning (its first attention
        # runs over the motif map), so the P4/P5 skip is never available there.
        skip_pc = bool(pair_cond is False and opt.SKIP_PAIR_COND
                       and not self.multi_motif)
        pair_condition = {"dmap": dmap, "omap": omap,
                          "dmap_mask": dmap_mask, "flags": flags}
        n = pos_noised.shape[0]
        # `uniform` is a (draws, N, N) stack, pre-drawn outside any CUDA graph.
        # Unbound ONCE - see `_unbind_draws`; slicing it per call cost 26
        # Dynamo graph breaks.
        draws = _unbind_draws(uniform, self.n_draws)


        _, _, local_pos = aa_frames_t(pos_noised)
        parts = [local_pos.reshape(n, -1),
                 GF.distance_rbf(vec_norm(local_pos), 0.0, 22.0).reshape(n, -1)]
        if self.time_embedding_on:
            parts = [self.time_embedding(t_pos)] + parts
        if self.diffuse_sequence:
            parts += [self.time_embedding(t_seq),
                      (seq_noised[..., None]
                       == torch.arange(21, device=seq_noised.device)).float()]
        parts.append(self.prepare_layer_norm_prev(prev_local))
        local = self.prepare_layer_norm_local(self.prepare_mlp(torch.cat(parts, dim=-1)))
        condition = self.prepare_layer_norm_cond(condition)

        if need_aux and need_aa:
            distogram_neighbours = get_random_neighbours(self.fape_neighbours)(
                pos_gt[:, 4], batch, mask, uniform=draws[0])
        else:
            distogram_neighbours = torch.zeros((pos_noised.shape[0], 0),
                                               dtype=torch.long,
                                               device=pos_noised.device)

        pos = pos_noised
        init_pos = prev_pos if self.multi_motif else None
        if self.preconditioning in ("edm", "edm_scaled"):
            pos_factors = edm_scaling(self.sigma_data, 1.0, alpha=1.0, beta=t_pos)
            if self.preconditioning == "edm_scaled":
                init_pos = self.sigma_data * pos_factors["in"][:, None, None] * pos
            pos = pos * pos_factors["skip"][:, None, None]

        per_block = 1 if self.multi_motif else 3
        # P4: what the pair blocks and the neighbour selector are allowed to
        # see. None means "there is no pair conditioning", which is a stronger
        # statement than "the tensors happen to be zero" - it is the caller's
        # assertion, checked at the call site.
        pc_used = None if skip_pc else pair_condition
        # P5: 16 -> 0 block-contact neighbours (K 64 -> 48 on this branch)
        num_block = 0 if skip_pc else 16

        # P6: everything in the two selectors that does not change between the
        # six blocks, computed once. `resi`/`chain`/`batch`/`mask` and
        # `prev_pos` are all loop invariants, so this is pure CSE - the tensors
        # it produces are the same bits the unhoisted code produced twelve and
        # six times respectively.
        hoist = opt.HOIST_NEIGHBOURS
        index_neighbours = cond_dist = cond_seed = None
        if hoist:
            index_neighbours = get_index_neighbours(16)(resi, chain, batch, mask)
            if not self.multi_motif:
                cond_dist, cond_seed = condition_neighbour_prologue(16, 16)(
                    prev_pos, pc_used, resi, chain, batch, mask,
                    index_neighbours=index_neighbours)

        def cond_neighbours_of(block_idx: int, cond_pos):
            base = 1 + block_idx * per_block
            if self.multi_motif:
                return extract_motif_neighbours(32)(pair_condition)
            if hoist:
                return condition_neighbour_tail(16, num_block)(
                    cond_dist, cond_seed, pc_used, batch, mask,
                    uniform=draws[base + 1], uniform2=draws[base + 2])
            return extract_condition_neighbours(
                16, 16, 16, num_block)(cond_pos, pc_used, resi, chain, batch,
                                       mask, uniform=draws[base + 1],
                                       uniform2=draws[base + 2])

        # P7: the condition branch depends only on `prev_pos` (a loop
        # invariant) and each block's condition neighbours, so all six blocks'
        # `pair_2` is one batched prologue. Its `cond_neighbours` have to be
        # drawn up front, which is why the selector is split out above - the
        # bits each block gets are unchanged, only the order they are computed
        # in moves.
        batch_cond = self.diffusion_stack.batched_cond_pair_available()
        cond_nbr_stack = None
        cond_pair = None
        if batch_cond:
            cond_nbr_stack = torch.stack(
                [cond_neighbours_of(i, prev_pos) for i in range(self.depth)])
            cond_pair = self.diffusion_stack.cond_pair_stack(
                prev_pos, pc_used, cond_nbr_stack, resi, chain, batch, mask,
                cyclic_mask=cyclic_mask, init_pos=init_pos)

        def neighbour_fn(cur_pos, cond_pos, block_idx: int = 0):
            base = 1 + block_idx * per_block
            current = extract_neighbours(16, 16, 32)(
                cur_pos, resi, chain, batch, mask, uniform=draws[base],
                index_neighbours=index_neighbours)
            cond = (cond_nbr_stack[block_idx] if cond_nbr_stack is not None
                    else cond_neighbours_of(block_idx, cond_pos))
            return current, cond

        local, pos, trajectory = self.diffusion_stack(
            local, pos, prev_pos, condition, pc_used, t_pos,
            resi, chain, batch, mask, neighbour_fn,
            cyclic_mask=cyclic_mask, init_pos=init_pos, cond_pair=cond_pair)

        if not need_aa:
            # P2. The three heads below are pure sinks on a non-output step.
            return (pos, local, trajectory)

        # NOT train_forward: its ADM masking draws RNG, and CUDA graphs cannot
        # capture an RNG op. The draw is a no-op at inference anyway - aa_gt is
        # all-MASK, so masking it to MASK changes nothing - which makes this
        # exactly equivalent while keeping the graph RNG-free.
        logits, aa_features = self.aa_decoder(aa_gt, local, pos, resi, chain, batch, mask)
        aa_logits = torch.log_softmax(logits, dim=-1)
        dssp = (torch.log_softmax(self.dssp_decoder(local), dim=-1) if need_aux
                else torch.zeros((local.shape[0], 3), dtype=local.dtype,
                                 device=local.device))
        aatype = aa_logits.argmax(dim=-1)
        raw_angles, angles, angle_pos = self.angle_head(aatype, local, pos)
        return (pos, local, trajectory, aa_logits, aa_features, dssp,
                distogram_neighbours, raw_angles, angles, angle_pos, aatype)

    # ------------------------------------------------------------------
    def forward(self, data: dict, prev: dict, predict_aa: bool = False,
                uniform: Tensor | None = None,
                compute_distogram: bool = True) -> dict:
        """One denoising pass.

        `uniform` optionally pins the Gumbel-top-k neighbour draw, which is the
        only nondeterminism at inference; supplying it makes the forward pass
        reproducible against the JAX reference.
        """
        c = self.config
        (local, pos, prev_pos, condition, pair_condition,
         t_pos, t_seq, resi, chain, batch, mask) = self.prepare_features(data, prev)

        # neighbours for distogram supervision, drawn off the *input* CB atoms
        distogram_neighbours = get_random_neighbours(c.fape_neighbours)(
            data["pos"][:, 4], batch, mask, uniform=uniform)

        cyclic_mask = data.get("cyclic_mask") if c.cyclic else None

        pos_factors = preconditioning_scale_factors(c, t_pos, c.sigma_data)
        init_pos = prev_pos if c.multi_motif else None
        if c.preconditioning == "edm":
            pos = pos * pos_factors["skip"][:, None, None]
        elif c.preconditioning == "edm_scaled":
            init_pos = c.sigma_data * pos_factors["in"][:, None, None] * pos
            pos = pos * pos_factors["skip"][:, None, None]

        # P6, same hoist as `forward_core` - kept here too so the parity tests,
        # which exercise this path, actually cover it.
        index_neighbours = cond_dist = cond_seed = None
        if opt.HOIST_NEIGHBOURS:
            index_neighbours = get_index_neighbours(16)(resi, chain, batch, mask)
            if not self.multi_motif:
                cond_dist, cond_seed = condition_neighbour_prologue(16, 16)(
                    prev_pos, pair_condition, resi, chain, batch, mask,
                    index_neighbours=index_neighbours)

        def cond_neighbours_of(cond_pos):
            if c.multi_motif:
                # the motif block's second neighbour set comes from the motif
                # distance map, not from self-conditioning geometry
                return extract_motif_neighbours(32)(pair_condition)
            if opt.HOIST_NEIGHBOURS:
                return condition_neighbour_tail(16, 16)(
                    cond_dist, cond_seed, pair_condition, batch, mask,
                    uniform=uniform)
            return extract_condition_neighbours(16, 16, 16, 16)(
                cond_pos, pair_condition, resi, chain, batch, mask,
                uniform=uniform)

        # P7, wired here as well as in `forward_core` so the four checkpoint
        # parity tests - which drive THIS path - actually cover it. The eager
        # path shares one `uniform` across the blocks, so the six condition
        # neighbour sets are identical; that changes nothing about the
        # prologue, which is indexed per block either way.
        cond_pair = None
        cond_nbr_stack = None
        if self.diffusion_stack.batched_cond_pair_available():
            cond_nbr_stack = torch.stack(
                [cond_neighbours_of(prev_pos) for _ in range(self.depth)])
            cond_pair = self.diffusion_stack.cond_pair_stack(
                prev_pos, pair_condition, cond_nbr_stack, resi, chain, batch,
                mask, cyclic_mask=cyclic_mask, init_pos=init_pos)

        def neighbour_fn(cur_pos, cond_pos, block_idx: int = 0):
            # the eager path takes a single (N, N) field (or None); the fast
            # path's per-draw stack is handled in forward_core
            current = extract_neighbours(16, 16, 32)(
                cur_pos, resi, chain, batch, mask, uniform=uniform,
                index_neighbours=index_neighbours)
            # MUST be the same index set the prologue built this block's pair
            # features from - `attn_2` gathers with it. Re-drawing here would
            # silently pair block b's features with block b's *other* random
            # neighbour set (with `uniform=None` the selector draws fresh
            # Gumbel noise on every call).
            cond = (cond_nbr_stack[block_idx] if cond_nbr_stack is not None
                    else cond_neighbours_of(cond_pos))
            return current, cond

        local, pos, trajectory = self.diffusion_stack(
            local, pos, prev_pos, condition, pair_condition, t_pos,
            resi, chain, batch, mask, neighbour_fn,
            cyclic_mask=cyclic_mask, init_pos=init_pos, cond_pair=cond_pair)

        result = {"trajectory": trajectory, "local": local, "pos": pos}

        aa_logits, decoder_features, corrupt_aa = self.aa_decoder.train_forward(
            data["aa_gt"], local, pos, resi, chain, batch, mask)
        result["aa"] = aa_logits
        result["aa_features"] = decoder_features
        result["corrupt_aa"] = corrupt_aa & data["mask"] & (data["aa_gt"] != 20)
        result["aa_gt"] = data["aa_gt"]
        result["pos_noised"] = data["pos_noised"]
        result["dssp"] = torch.log_softmax(self.dssp_decoder(local), dim=-1)

        result["distogram_neighbours"] = distogram_neighbours
        if compute_distogram:
            result["distogram"] = self.distogram_decoder(
                local, pos, distogram_neighbours, resi, chain, batch, mask,
                cyclic_mask=cyclic_mask)

        aatype = data["aa_gt"]
        if predict_aa:
            aatype = aa_logits.argmax(dim=-1)
        if "aa_condition" in data:
            aatype = torch.where(data["aa_condition"] != 20, data["aa_condition"], aatype)
        result["aatype"] = aatype

        raw_angles, angles, angle_pos = self.angle_head(aatype, local, pos)
        result["raw_angles"] = raw_angles
        result["angles"] = angles
        result["atom_pos"] = pos if c.encode_atom14 else angle_pos
        return result
