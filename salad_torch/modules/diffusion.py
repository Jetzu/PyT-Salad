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

from ..diffusion.precond import preconditioning_scale_factors
from ..geometry import features as GF
from ..geometry.frames import extract_aa_frames
from ..geometry.neighbours import (extract_neighbours, get_index_neighbours,
                                   get_random_neighbours, get_spatial_neighbours)
from ..geometry.rigid import vec_norm
from .basic import MLP, HkLayerNorm, HkLinear, gelu
from ..config import uses_init_pos
from .blocks import (DiffusionStack, extract_condition_neighbours,
                     extract_motif_neighbours)
from .decoders import AADecoder, AngleHead, DistogramDecoder

__all__ = ["Diffusion", "uniform_draw_count"]

class _UniformPool:
    """Hands out one pre-drawn ``(N, N)`` uniform field per random draw.

    CUDA graphs cannot capture an RNG op ("Offset increment outside graph
    capture"), so every Gumbel-top-k draw has to be hoisted out of the graph.
    The draws happen in a fixed order, so slicing a stacked tensor reproduces
    them exactly while keeping the graph RNG-free. Pass `None` to fall back to
    drawing inline (eager only).
    """

    __slots__ = ("stack", "i")

    def __init__(self, stack):
        self.stack = stack
        self.i = 0

    def next(self):
        if self.stack is None:
            return None
        u = self.stack[self.i]
        self.i += 1
        return u


def uniform_draw_count(config) -> int:
    """How many (N, N) uniform fields one forward consumes.

    1 for the distogram neighbours, then per block: 1 for extract_neighbours
    and, unless multi_motif routes to the motif map, 2 for
    extract_condition_neighbours (its random and block-contact stages).
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

    # ------------------------------------------------------------------
    def time_embedding(self, t: Tensor) -> Tensor:
        if self.config.diffusion_kind in self._VP_KINDS:
            return vp_time_embedding(t)
        return edm_time_embedding(t)

    def prepare_features(self, data: dict, prev: dict) -> tuple:
        c = self.config
        pos = data["pos_noised"]
        t_pos, t_seq = data["t_pos"], data["t_seq"]
        seq = data["seq_noised"]
        n = pos.shape[0]

        _, local_pos = extract_aa_frames(pos)
        parts = [
            local_pos.reshape(n, -1),
            GF.distance_rbf(vec_norm(local_pos), 0.0, 22.0).reshape(n, -1),
        ]
        if c.time_embedding:
            parts = [self.time_embedding(t_pos)] + parts
        if c.diffuse_sequence:
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
                     ) -> tuple[Tensor, ...]:
        """Returns (pos, local, trajectory, aa_logits, aa_features, dssp,
        distogram_neighbours, raw_angles, angles, angle_pos, aatype)."""
        c = self.config
        pair_condition = {"dmap": dmap, "omap": omap,
                          "dmap_mask": dmap_mask, "flags": flags}
        n = pos_noised.shape[0]
        # `uniform` is a (draws, N, N) stack, pre-drawn outside any CUDA graph
        pool = _UniformPool(uniform)

        _, local_pos = extract_aa_frames(pos_noised)
        parts = [local_pos.reshape(n, -1),
                 GF.distance_rbf(vec_norm(local_pos), 0.0, 22.0).reshape(n, -1)]
        if c.time_embedding:
            parts = [self.time_embedding(t_pos)] + parts
        if c.diffuse_sequence:
            parts += [self.time_embedding(t_seq),
                      (seq_noised[..., None]
                       == torch.arange(21, device=seq_noised.device)).float()]
        parts.append(self.prepare_layer_norm_prev(prev_local))
        local = self.prepare_layer_norm_local(self.prepare_mlp(torch.cat(parts, dim=-1)))
        condition = self.prepare_layer_norm_cond(condition)

        distogram_neighbours = get_random_neighbours(c.fape_neighbours)(
            pos_gt[:, 4], batch, mask, uniform=pool.next())

        pos = pos_noised
        pos_factors = preconditioning_scale_factors(c, t_pos, c.sigma_data)
        init_pos = prev_pos if c.multi_motif else None
        if c.preconditioning == "edm":
            pos = pos * pos_factors["skip"][:, None, None]
        elif c.preconditioning == "edm_scaled":
            init_pos = c.sigma_data * pos_factors["in"][:, None, None] * pos
            pos = pos * pos_factors["skip"][:, None, None]

        def neighbour_fn(cur_pos, cond_pos):
            current = extract_neighbours(16, 16, 32)(
                cur_pos, resi, chain, batch, mask, uniform=pool.next())
            if c.multi_motif:
                cond = extract_motif_neighbours(32)(pair_condition)
            else:
                cond = extract_condition_neighbours(
                    16, 16, 16, 16)(cond_pos, pair_condition, resi, chain, batch,
                                    mask, uniform=pool.next(), uniform2=pool.next())
            return current, cond

        local, pos, trajectory = self.diffusion_stack(
            local, pos, prev_pos, condition, pair_condition, t_pos,
            resi, chain, batch, mask, neighbour_fn,
            cyclic_mask=cyclic_mask, init_pos=init_pos)

        # NOT train_forward: its ADM masking draws RNG, and CUDA graphs cannot
        # capture an RNG op. The draw is a no-op at inference anyway - aa_gt is
        # all-MASK, so masking it to MASK changes nothing - which makes this
        # exactly equivalent while keeping the graph RNG-free.
        logits, aa_features = self.aa_decoder(aa_gt, local, pos, resi, chain, batch, mask)
        aa_logits = torch.log_softmax(logits, dim=-1)
        dssp = torch.log_softmax(self.dssp_decoder(local), dim=-1)
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

        def neighbour_fn(cur_pos, cond_pos):
            current = extract_neighbours(16, 16, 32)(
                cur_pos, resi, chain, batch, mask, uniform=uniform)
            if c.multi_motif:
                # the motif block's second neighbour set comes from the motif
                # distance map, not from self-conditioning geometry
                cond = extract_motif_neighbours(32)(pair_condition)
            else:
                cond = extract_condition_neighbours(16, 16, 16, 16)(
                    cond_pos, pair_condition, resi, chain, batch, mask,
                    uniform=uniform)
            return current, cond

        local, pos, trajectory = self.diffusion_stack(
            local, pos, prev_pos, condition, pair_condition, t_pos,
            resi, chain, batch, mask, neighbour_fn,
            cyclic_mask=cyclic_mask, init_pos=init_pos)

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
