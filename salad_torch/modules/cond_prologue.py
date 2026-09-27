"""P7: the condition branch's pair features for all `diffusion_depth` blocks,
computed once as a batched prologue.

Why this is legal
-----------------
`DiffusionBlock.pair_2` reads `prev_pos` - the *same tensor for all six blocks*
- at that block's `cond_neighbours`. Nothing in it depends on `features`, on
`pos`, or on any earlier block's output; only the weights differ per block. So
the whole stage is a prologue over six independent, identically-shaped
problems, i.e. one batched problem.

What it costs numerically
-------------------------
The feature builders are run on a **tiled** residue axis (`prev_pos` repeated
`depth` times, with per-copy chain/batch offsets so the copies cannot see each
other and the ``-1`` sentinel wraps *within* its copy). Every one of them is
elementwise or per-residue, so the feature tensors are bit-identical to the
per-block ones. The single change is that the seven projections and the two
MLP layers become `bmm(batch=depth)` instead of `depth` separate `mm`s, which
may pick a different tile or split-K: **reassociation only**, gated like P5.

One assumption, and it only bites a CYCLIC config: the tiling offsets the
chain/batch ids by `b * N`, so dense ids (always `< N`, which `unique_chain`
and the CLI guarantee) keep the copies disjoint. Nothing else depends on it -
no pair ever crosses a copy, because the neighbour indices are offset into
their own copy first - but the cyclic branch takes a GLOBAL `index_count` over
the chain ids, which would merge two copies if their offset ids collided.

Not available for `multi_motif`: `MotifAdapterDiffusionBlock.pair_2` runs over
`pos`, which changes every block, so that path stays sequential.

Staleness
---------
The stacked weights are a **cache** of the live per-block parameters, refreshed
by a `load_state_dict` post-hook and by an explicit `sync()`. The batched path
is therefore only taken in `eval()` mode - a training step mutates the
parameters without touching the cache.
"""
from __future__ import annotations

import os
import torch
from torch import Tensor, nn

from ..geometry import features as GF
from .basic import gelu

__all__ = ["StackedCondPair"]

#: the projections `DiffusionPairFeatures.forward` applies, in the order it
#: adds them. `dmap`/`flags` are only used when there IS pair conditioning.
_PROJ = ("relpos", "dmap", "flags", "dist", "dir", "rot", "pairvec")


#: How to stop inductor fusing the neighbour-index producer into its consumers.
#: "clone"  - cheapest, may be elided by inductor
#: "custom" - an opaque custom op, cannot be fused through
#: "break"  - `torch._dynamo.graph_break()`, always works, splits the graph
#: "none"   - no barrier (reproduces the miscompile)
_P7_BARRIER = os.environ.get("SALAD_P7_BARRIER", "custom").strip().lower()

try:
    @torch.library.custom_op("salad::p7_barrier", mutates_args=())
    def _p7_barrier_op(x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @_p7_barrier_op.register_fake
    def _(x):
        return torch.empty_like(x)
except Exception:  # pragma: no cover - older torch without custom_op
    _p7_barrier_op = None


def _p7_barrier(x: torch.Tensor) -> torch.Tensor:
    if _P7_BARRIER == "none":
        return x
    if _P7_BARRIER == "clone":
        return x.clone()
    if _P7_BARRIER == "break":
        torch._dynamo.graph_break()
        return x
    if _p7_barrier_op is not None:
        return _p7_barrier_op(x)
    return x.clone()


class StackedCondPair(nn.Module):
    """`pair_2` for every block of a `DiffusionStack`, in one batched pass."""

    def __init__(self, blocks, pair_size: int, use_init_pos: bool = False,
                 cyclic: bool = False):
        super().__init__()
        self.depth = len(blocks)
        self.pair_size = pair_size
        self.use_init_pos = bool(use_init_pos)
        #: recorded for introspection ONLY. `forward` deliberately does not
        #: read it - see the `cyclic=cyc_t is not None` note there.
        self.cyclic = bool(cyclic)
        self._names = list(_PROJ) + (["init_dist", "init_rot"]
                                     if self.use_init_pos else [])
        # NB the blocks are NOT kept: `sync`/`in_sync` are handed the live
        # `nn.ModuleList` by the owner, so a deep copy of the stack can never
        # end up refreshing its cache from the ORIGINAL model's parameters.
        for name, t in self._stacked(blocks).items():
            self.register_buffer(name, t, persistent=False)
        self.ln_eps = float(blocks[0].pair_2.layer_norm.eps)

    # -- the weight cache ------------------------------------------------
    def _stacked(self, blocks) -> dict[str, Tensor]:
        mods = [b.pair_2 for b in blocks]
        st = {f"w_{n}": torch.stack([getattr(m, f"linear_{n}").w.detach()
                                     for m in mods]).contiguous()
              for n in self._names}
        for i in (0, 1):
            st[f"w_mlp{i}"] = torch.stack(
                [m.mlp.layers[i].w.detach() for m in mods]).contiguous()
            st[f"b_mlp{i}"] = torch.stack(
                [m.mlp.layers[i].b.detach() for m in mods]).contiguous()
        st["ln_scale"] = torch.stack(
            [m.layer_norm.scale.detach() for m in mods]).contiguous()
        st["ln_offset"] = torch.stack(
            [m.layer_norm.offset.detach() for m in mods]).contiguous()
        return st

    @torch.no_grad()
    def sync(self, blocks) -> "StackedCondPair":
        """Re-read the per-block parameters. Call after any weight load."""
        for name, t in self._stacked(blocks).items():
            getattr(self, name).copy_(t)
        return self

    def in_sync(self, blocks) -> bool:
        """True when the cache still matches the live parameters, bit for bit."""
        try:
            return all(torch.equal(getattr(self, name), t)
                       for name, t in self._stacked(blocks).items())
        except RuntimeError:                       # shape/device mismatch
            return False

    # -- the prologue ----------------------------------------------------
    def forward(self, prev_pos: Tensor, pair_condition: dict | None,
                neighbours: Tensor, resi: Tensor, chain: Tensor, batch: Tensor,
                mask: Tensor, cyclic_mask: Tensor | None = None,
                init_pos: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """`neighbours` is the **stacked** ``(depth, N, K)`` condition index."""
        b, n, k = neighbours.shape
        mask = mask.bool()
        pair_mask = mask[None, :, None] & mask[neighbours] & (neighbours != -1)

        # Tile the residue axis. `chain`/`batch` get a `b * n` offset, which is
        # safe because a dense chain id is always < n, so the copies are
        # mutually "different chain" / "different batch" and can never form a
        # cross-copy relative position. The `-1` sentinel is mapped to the LAST
        # residue OF ITS OWN COPY, which is exactly the row NumPy-style
        # negative indexing picks in the untiled code - so the garbage lanes
        # (which `pair_mask` discards) hold the same garbage.
        off = (torch.arange(b, device=neighbours.device) * n)[:, None, None]
        nb = torch.where(neighbours >= 0, neighbours + off,
                         off + (n - 1)).reshape(b * n, k)
        # P7 CORRECTNESS BARRIER - do not remove without reading opt.BATCH_COND.
        # PyTorch 2.5.1's inductor fuses the op that PRODUCES this (600, 48)
        # index tensor into the indirect gathers that CONSUME it, and emits an
        # impossible load mask: `tmp36 & tmp4` is `(row >= 100) & (row < 100)`,
        # false for every row, so all three `tl.load`s return their `other=0.0`
        # fallback and the relative-position bins are computed from zeros.
        # 23,070 of 28,800 bins wrong, max error 64 bins; downstream that is
        # `pos` off by 9.586 A and 17-21 % of designs destroyed.
        # `_P7_BARRIER` selects how we stop that fusion; see opt.BATCH_COND.
        nb = _p7_barrier(nb)
        pos_t = prev_pos.repeat(b, 1, 1)
        resi_t = resi.repeat(b)
        chain_t = (chain[None] + off[:, :, 0]).reshape(-1)
        batch_t = (batch[None] + off[:, :, 0]).reshape(-1)
        cyc_t = None if cyclic_mask is None else cyclic_mask.repeat(b)

        feats: dict[str, Tensor] = {}
        # `cyclic` is taken from the ARGUMENT, exactly as `pair_2` takes it
        # (`blocks.DiffusionPairFeatures.forward`: `cyclic=cyclic_mask is not
        # None`) - NOT from `config.cyclic`. The two disagree whenever a cyclic
        # config is called without a `cyclic_mask`, which both entry points
        # allow (`diffusion.py` and `fast.py` both do `data.get("cyclic_mask")
        # if config.cyclic else None`, and `.get` returns None on a key that is
        # simply absent). Reading the config here would make the prologue wrap
        # relative positions that the sequential path leaves unwrapped - a
        # WRONG ANSWER, not a reassociation. Mirror the predicate instead.
        relpos = GF.sequence_relative_position(
            32, one_hot=False, cyclic=cyc_t is not None)(
                resi_t, chain_t, batch_t, nb, cyclic_mask=cyc_t)
        # P3's row gather, batched: W is (depth, 66, P), relpos is (depth, N, K)
        relpos = relpos.long().reshape(b, n * k)

        if pair_condition is not None:
            idx = torch.arange(n, device=prev_pos.device)
            g = {key: torch.stack([pair_condition[key][idx[:, None], neighbours[i]]
                                   for i in range(b)])
                 for key in ("dmap", "dmap_mask", "flags")}
            feats["dmap"] = torch.where(
                g["dmap_mask"][..., None], GF.distance_rbf(g["dmap"]),
                torch.zeros((), device=prev_pos.device, dtype=prev_pos.dtype))
            flags = g["flags"]
            feats["flags"] = torch.where(flags.any(dim=-1)[..., None], flags,
                                         torch.zeros_like(flags))
        feats["dist"] = GF.distance_features(pos_t, nb, 0.0, 22.0)
        feats["dir"] = GF.direction_features(pos_t, nb)
        feats["rot"] = GF.position_rotation_features(pos_t, nb)
        feats["pairvec"] = GF.pair_vector_features(pos_t, nb)
        if self.use_init_pos:
            if init_pos is None:
                raise ValueError("use_init_pos=True but init_pos was not supplied")
            ipos_t = init_pos.repeat(b, 1, 1)
            feats["init_dist"] = GF.distance_features(ipos_t, nb, 0.0, 22.0)
            feats["init_rot"] = GF.position_rotation_features(ipos_t, nb)

        # P3's row gather, per block, out of the stacked (depth, 66, P) weight.
        # Advanced indexing, not `torch.gather`: gather would need an
        # (depth, N*K, P) int64 index, i.e. 15 MB of pure bookkeeping at L=100.
        ar_b = torch.arange(b, device=prev_pos.device)[:, None]
        pair = self.w_relpos[ar_b, relpos]
        for name in self._names:
            if name == "relpos":
                continue
            x = feats.get(name)
            if x is None:                            # dmap/flags with no pair cond
                continue
            pair = pair + torch.bmm(x.reshape(b, n * k, x.shape[-1]),
                                    getattr(self, f"w_{name}"))

        mu = pair.mean(-1, keepdim=True)
        var = pair.var(-1, unbiased=False, keepdim=True)
        pair = (pair - mu) * torch.rsqrt(var + self.ln_eps)
        pair = pair * self.ln_scale[:, None] + self.ln_offset[:, None]
        pair = gelu(torch.bmm(pair, self.w_mlp0) + self.b_mlp0[:, None])
        pair = torch.bmm(pair, self.w_mlp1) + self.b_mlp1[:, None]
        return pair.reshape(b, n, k, self.pair_size), pair_mask
