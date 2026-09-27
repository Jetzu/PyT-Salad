"""Sparse invariant point attention - the core of the salad denoiser.

Shapes throughout: ``N`` residues, ``K`` neighbours, ``H`` heads, ``S`` key
size, ``QP``/``VP`` query/value points, ``P`` pair channels.

Invariance: q/k/v points are generated in each residue's local frame and pushed
to global coordinates, so ``|q_g_i - k_g_j|`` is invariant; the aggregated value
points are pulled back with ``apply_inverse_to_point``, making the output
invariant too. Reflections are not handled - frames are proper rotations by
construction.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .. import opt
from ..geometry.frames import aa_frames_t, apply_inverse_t, apply_t
from ..geometry.rigid import Rigid, vec_norm
from .basic import HkLayerNorm, HkLinear

__all__ = ["LinearToPoints", "SparseInvariantPointAttention", "SparseStructureAttention"]


class LinearToPoints(nn.Module):
    """Project features to ``size`` points, expressed in the global frame."""

    def __init__(self, in_size: int, size: int = 8, init: str = "linear"):
        super().__init__()
        self.size = size
        self.linear = HkLinear(in_size, size * 3, bias=False, initializer=init)

    def forward(self, data: Tensor, rot: Tensor, trans: Tensor) -> Tensor:
        # JAX computes points in fp32 then casts back to the input dtype
        # geometry is done at fp32 or better; under autocast this keeps
        # precision, and it lets a float64 model actually run.
        dtype = data.dtype if data.dtype in (torch.float32, torch.float64) else torch.float32
        raw = self.linear(data).to(dtype)
        points = raw.reshape(*raw.shape[:-1], self.size, 3)
        out = apply_t(rot, trans, points, k=1)
        return out.to(data.dtype)


class SparseInvariantPointAttention(nn.Module):
    """Haiku scope ``ada_point_attention``."""

    def __init__(self, local_size: int, pair_size: int, size: int = 32, heads: int = 4,
                 query_points: int = 8, value_points: int = 8,
                 final_init: str = "zeros", normalize: bool = False):
        super().__init__()
        self.size = size
        self.heads = heads
        self.query_points = query_points
        self.value_points = value_points
        self.normalize = normalize

        # Haiku numbers LayerNorms by creation order within this scope, so the
        # q/k norms are layer_norm_1/_2 only because the `normalize` norm took
        # layer_norm. With normalize=False they would shift down to
        # layer_norm/_1 and the checkpoint map would silently mis-bind.
        # Every shipped checkpoint has normalize=True (its ada_point_attention
        # layer_norm is local_size-wide), so refuse the other case rather than
        # mis-map it.
        if not normalize:
            raise NotImplementedError(
                "normalize=False shifts the Haiku LayerNorm indices "
                "(q/k become layer_norm/layer_norm_1); no shipped salad "
                "checkpoint uses it, so the name map is not defined here")
        self.layer_norm = HkLayerNorm(local_size)
        self.qkv = HkLinear(local_size, heads * 3 * size, bias=False)
        # separate LayerNorms on q and k, over the key-size axis
        self.layer_norm_1 = HkLayerNorm(size)
        self.layer_norm_2 = HkLayerNorm(size)
        n_points = heads * (2 * query_points + value_points)
        self.qkv_global = LinearToPoints(local_size, n_points)
        self.bias = HkLinear(pair_size, heads, bias=False)
        self.gamma = nn.Parameter(torch.full((heads,), math.log(math.e - 1.0)))
        out_width = heads * pair_size + heads * size + heads * value_points * 3\
            + heads * value_points
        self.project_out = HkLinear(out_width, local_size, initializer=final_init)

    def forward(self, local: Tensor, pair: Tensor, rot: Tensor, trans: Tensor,
                neighbours: Tensor, mask: Tensor) -> Tensor:
        local = self.layer_norm(local)
        n = local.shape[0]
        h, s = self.heads, self.size
        qp, vp = self.query_points, self.value_points

        qkv = self.qkv(local).reshape(n, h, 3 * s)
        q, k, v = torch.split(qkv, s, dim=-1)
        q = self.layer_norm_1(q)
        k = self.layer_norm_2(k)

        qkv_g = self.qkv_global(local, rot, trans)            # (N, H*(2QP+VP), 3)
        qkv_g = qkv_g.reshape(n, h, 2 * qp + vp, 3)
        q_g = qkv_g[:, :, :qp]
        k_g = qkv_g[:, :, qp:2 * qp]
        v_g = qkv_g[:, :, 2 * qp:]

        bias = self.bias(pair)                             # (N, K, H)

        w_c = math.sqrt(2.0 / (9.0 * qp))
        w_l = math.sqrt(1.0 / 3.0)
        dfactor = nn.functional.softplus(self.gamma).reshape(1, 1, h) * w_c / 2

        # (N,1,H,QP,3) - (N,K,H,QP,3) -> (N,K,H)
        dist = dfactor * ((q_g[:, None] - k_g[neighbours]) ** 2).sum(dim=(-1, -2))
        if opt.IPA_MULSUM:
            # O2: read the gather in place instead of cloning it into
            # `bmm` layout. Reassociation only - see `opt.IPA_MULSUM`.
            dot = math.sqrt(1.0 / s) * (q[:, None] * k[neighbours]).sum(-1)
        else:
            dot = math.sqrt(1.0 / s) * torch.einsum("ihc,ijhc->ijh", q, k[neighbours])
        logits = w_l * (dot + bias - dist)

        pair_mask = mask.bool() & (neighbours != -1)
        logits = torch.where(pair_mask[..., None], logits,
                             torch.full_like(logits, -1e9))
        attn = torch.softmax(logits, dim=-2)               # over K
        # second masking after softmax: fully-masked rows would otherwise be garbage
        attn = torch.where(pair_mask[..., None], attn, torch.zeros_like(attn))

        if opt.IPA_MULSUM:
            out_pair = (attn[..., None] * pair[:, :, None, :]).sum(1)
            out_scalar = (attn[..., None] * v[neighbours]).sum(1)
            out_point = (attn[..., None, None] * v_g[neighbours]).sum(1)
        else:
            out_pair = torch.einsum("ijh,ijc->ihc", attn, pair)
            out_scalar = torch.einsum("ijh,ijhc->ihc", attn, v[neighbours])
            out_point = torch.einsum("ijh,ijhpc->ihpc", attn, v_g[neighbours])
        out_point = apply_inverse_t(rot, trans, out_point.to(trans.dtype), k=2)
        out_norm = vec_norm(out_point)

        out = self.project_out(torch.cat((
            out_pair.reshape(n, -1),
            out_scalar.reshape(n, -1),
            out_point.reshape(n, -1),
            out_norm.reshape(n, -1),
        ), dim=-1))
        return out.to(local.dtype)


class SparseStructureAttention(nn.Module):
    """Thin wrapper that derives frames from positions, Haiku scope
    ``sparse_structure_attn``."""

    def __init__(self, config, local_size: int, pair_size: int, normalize: bool = True):
        super().__init__()
        final_init = config.update_init if config.update_init else "zeros"
        self.ada_point_attention = SparseInvariantPointAttention(
            local_size=local_size, pair_size=pair_size,
            heads=config.heads, size=config.key_size,
            final_init=final_init, normalize=normalize)

    def forward(self, local: Tensor, pos: Tensor, pair: Tensor, pair_mask: Tensor,
                neighbours: Tensor) -> Tensor:
        rot, trans, _ = aa_frames_t(pos)
        return self.ada_point_attention(local, pair, rot, trans, neighbours, pair_mask)
