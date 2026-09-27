"""Distance binning, relative-position encodings and pair features."""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .rigid import Rigid, vec_norm, vec_normalized
from .frames import aa_frames_t, apply_inverse_t, extract_aa_frames
from .segment import index_count

__all__ = [
    "linear_centers", "gaussian_rbf", "distance_rbf", "distance_one_hot",
    "fourier_time_embedding", "sequence_relative_position",
    "distance_features", "direction_features", "rotation_features",
    "position_rotation_features", "pair_vector_features",
    "direction_and_pair_vector_features",
]


# --------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------
def linear_centers(min_val: float = 0.0, max_val: float = 22.0, bins: int = 64,
                   device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """Bin centres and per-bin widths, matching `geometry.linear_centers`.

    Note this is ``linspace`` (endpoint inclusive), and the width array repeats
    its first element so it has the same length as `centers`.
    """
    centers = torch.linspace(min_val, max_val, bins, device=device, dtype=dtype)
    step = (centers[1:] - centers[:-1]).abs()
    step = torch.cat((step[:1], step), dim=0)
    return centers, step


def gaussian_rbf(x: Tensor, centers: Tensor, step: Tensor) -> Tensor:
    """``exp(-(x[..., None] - c)^2 / step^2)``.

    Two upstream quirks preserved: the trailing axis is added *inside* this
    function, and the denominator is ``step**2``, not ``2*sigma**2``.
    """
    return torch.exp(-((x[..., None] - centers) ** 2) / step ** 2)


def _arange_centers(min_distance: float, max_distance: float, bins: int,
                    device=None, dtype=torch.float32) -> tuple[Tensor, float]:
    """`distance_rbf`'s *default* centres - note these are NOT `linear_centers`.

    ``step = (max - min) / bins`` (a scalar) and
    ``centers = min + arange(bins) * step + step / 2``, i.e. bin midpoints, which
    differ from the endpoint-inclusive ``linspace`` used by `linear_centers`.
    """
    step = (max_distance - min_distance) / bins
    centers = (min_distance
               + torch.arange(bins, device=device, dtype=dtype) * step
               + step / 2)
    return centers, step


def distance_rbf(distance: Tensor, min_distance: float = 0.0,
                 max_distance: float = 22.0, bins: int = 64,
                 centers: Tensor | None = None) -> Tensor:
    """RBF-expand a distance tensor to ``(..., bins)``.

    With explicit `centers`, the per-bin width is derived from their spacing
    (as upstream does); otherwise the scalar-step midpoint grid is used.
    """
    if centers is None:
        c, step_v = _arange_centers(min_distance, max_distance, bins,
                                    device=distance.device, dtype=distance.dtype)
        step: Tensor | float = step_v
    else:
        c = centers
        step = (c[1:] - c[:-1]).abs()
        step = torch.cat((step[:1], step), dim=0)
    return gaussian_rbf(distance, c, step)


def distance_one_hot(distance: Tensor, min_distance: float = 0.0,
                     max_distance: float = 22.0, bins: int = 64) -> Tensor:
    """Hard binning onto the same midpoint grid: nearest centre gets 1."""
    centers, _ = _arange_centers(min_distance, max_distance, bins,
                                 device=distance.device, dtype=distance.dtype)
    idx = (distance[..., None] - centers).abs().argmin(dim=-1)
    return torch.nn.functional.one_hot(idx, bins).to(distance.dtype)


def fourier_time_embedding(t: Tensor, size: int = 128, base: float = 10_000.0,
                           scale: float = 1.0) -> Tensor:
    """``concat(sin(scale*t / base**e), cos(...))`` -> ``(..., size)``."""
    exponent = 2.0 * torch.arange(size // 2, device=t.device, dtype=t.dtype) / size
    denom = base ** exponent
    arg = scale * t[..., None] / denom
    return torch.cat((torch.sin(arg), torch.cos(arg)), dim=-1)


# --------------------------------------------------------------------------
# relative sequence position
# --------------------------------------------------------------------------
def sequence_relative_position(count: int = 32, one_hot: bool = True,
                               cyclic: bool = False, identify_ends: bool = False,
                               pseudo_chains: bool = False):
    """Relative residue-index encoding.

    Bins: ``clip(resi_i - resi_j, -count, count) + count`` in ``[0, 2*count]``,
    with cross-chain / cross-batch pairs placed in the extra bin ``2*count+1``.
    ``count_total = 2*count + 2`` (= 66 for count=32).

    Note the default of `one_hot` is flipped relative to upstream (which
    defaults to False); every salad call site passes it explicitly.

    `identify_ends` and `pseudo_chains` remap the bins entirely and are unused
    by the noise_schedule_benchmark family, so they are refused rather than
    silently ignored.
    """
    if identify_ends or pseudo_chains:
        raise NotImplementedError(
            "identify_ends / pseudo_chains change count_total and the bin "
            "layout; they are unused by the salad diffusion models and are not "
            "ported. Implement them before enabling.")
    count_total = 2 * count + 2

    def inner(resi: Tensor, chain: Tensor, batch: Tensor,
              neighbours: Tensor | None = None,
              cyclic_mask: Tensor | None = None) -> Tensor:
        if neighbours is None:
            same_chain = chain[:, None] == chain[None, :]
            same_batch = batch[:, None] == batch[None, :]
            rel = resi[:, None] - resi[None, :]
        else:
            same_chain = chain[:, None] == chain[neighbours]
            same_batch = batch[:, None] == batch[neighbours]
            rel = resi[:, None] - resi[neighbours]

        if cyclic:
            # NB: upstream counts chain lengths with an all-ones mask, so the
            # real residue mask never shrinks a chain (and never divides by 0).
            lengths = index_count(chain, torch.ones_like(chain, dtype=torch.bool))
            length = lengths[:, None].to(rel.dtype)
            wrap = rel.abs() > length / 2
            if cyclic_mask is not None:
                wrap = wrap & cyclic_mask[:, None]
            wrapped = torch.where(rel < 0,
                                  torch.remainder(rel, length),
                                  torch.remainder(rel, length) - length)
            rel = torch.where(wrap, wrapped, rel)

        binned = rel.clamp(-count, count) + count
        other = torch.full_like(binned, 2 * count + 1)
        binned = torch.where(same_chain, binned, other)
        binned = torch.where(same_batch, binned, other)
        if not one_hot:
            return binned
        return torch.nn.functional.one_hot(binned.long(), count_total).to(torch.float32)

    return inner


# --------------------------------------------------------------------------
# pair features
# --------------------------------------------------------------------------
def distance_features(pos: Tensor, neighbours: Tensor, d_min: float = 0.0,
                      d_max: float = 22.0, num_rbf: int = 16) -> Tensor:
    """Pairwise distances between the first 5 atoms -> ``(N, K, 5*5*num_rbf)``."""
    # (N, 1, 5, 1, 3) - (N, K, 1, 5, 3) -> (N, K, 5, 5)
    left = pos[:, None, :5, None, :]
    right = pos[neighbours][:, :, None, :5, :]
    dist = vec_norm(left - right)
    n, k = dist.shape[0], dist.shape[1]
    dist = dist.reshape(n, k, 25)
    return distance_rbf(dist, d_min, d_max, num_rbf).reshape(n, k, 25 * num_rbf)


def direction_features(pos: Tensor, neighbours: Tensor) -> Tensor:
    """Unit vectors to neighbour atoms, in the residue's own frame -> ``(N, K, 3A)``."""
    rot, trans, _ = aa_frames_t(pos)
    # neighbour atom positions expressed in frame i
    neighbour_pos = pos[neighbours]                       # (N, K, A, 3)
    local = apply_inverse_t(rot, trans, neighbour_pos, k=2)
    local = vec_normalized(local)
    n, k, a, _ = local.shape
    return local.reshape(n, k, a * 3)


def rotation_features(frames: Rigid, neighbours: Tensor) -> Tensor:
    """Relative rotation between frames -> ``(N, K, 9)``."""
    rot_i = frames.rotation[:, None]                      # (N, 1, 3, 3)
    rot_j = frames.rotation[neighbours]                   # (N, K, 3, 3)
    rel = rot_i.transpose(-1, -2) @ rot_j
    return rel.reshape(rel.shape[0], rel.shape[1], 9)


def position_rotation_features(pos: Tensor, neighbours: Tensor) -> Tensor:
    rot, _, _ = aa_frames_t(pos)
    rel = rot[:, None].transpose(-1, -2) @ rot[neighbours]
    return rel.reshape(rel.shape[0], rel.shape[1], 9)


def pair_vector_features(pos: Tensor, neighbours: Tensor, scale: float = 0.1) -> Tensor:
    """Directions and scaled coordinates of both residues' atoms in frame i.

    -> ``(N, K, 12*A)``: ``[dir(2A*3), scale*coord(2A*3)]``.
    """
    rot, trans, _ = aa_frames_t(pos)
    n, a, _ = pos.shape
    k = neighbours.shape[1]
    self_pos = pos[:, None].expand(n, k, a, 3)            # (N, K, A, 3)
    neigh_pos = pos[neighbours]                           # (N, K, A, 3)
    both = torch.cat([self_pos, neigh_pos], dim=-2)       # (N, K, 2A, 3)
    local = apply_inverse_t(rot, trans, both, k=2)
    directions = vec_normalized(local).reshape(n, k, 2 * a * 3)
    coords = (scale * local).reshape(n, k, 2 * a * 3)
    return torch.cat([directions, coords], dim=-1)


def direction_and_pair_vector_features(pos: Tensor, neighbours: Tensor,
                                       scale: float = 0.1) -> tuple[Tensor, Tensor]:
    """N028/O6: `direction_features` and `pair_vector_features` sharing work.

    `DiffusionPairFeatures.forward` calls both on the SAME `(pos, neighbours)`,
    and both start by computing `aa_frames_t(pos)` and gathering
    `pos[neighbours]`. Sharing THOSE two - and calling each function's own
    `apply_inverse_t`/`vec_normalized` exactly once, at exactly the shape it
    already uses - removes one frame computation and one neighbour gather.

    `fx_graph_cse` (N017/O1) cannot see this on its own: `pair_vector_features`
    feeds `apply_inverse_t` a `cat([self_pos, neigh_pos])`, so its subgraph is
    not syntactically equal to `direction_features`'s, even though the first
    A rows of one computation reuse values the other also needs.

    BIT-EXACT, measured not just argued: a neighbour gather has no rounding,
    so reusing `neigh_pos` is free. But do NOT go further and also share the
    `apply_inverse_t`/`vec_normalized` CALLS themselves (e.g. transforming
    `self_pos` and `neigh_pos` separately and concatenating the transformed
    halves, or transforming `cat([self_pos, neigh_pos])` once and slicing the
    neighbour half back out for `direction`) - every such reshuffle was tried
    and DIFFERS from today's shipped output by ~1e-7 starting at diffusion
    block 2, but ONLY inside a live model forward (a fresh, isolated call on
    the exact same captured tensors reproduces bit-for-bit). This is eager
    CPU PyTorch's batched-linalg lowering picking a different low-level
    kernel/rounding for the SAME math depending on the surrounding call
    context, not a bug in the arithmetic - see RESULTS.md section 3 for the
    two-process, block-by-block trace that isolated it. The only variant
    verified safe is this one: each of the two functions' own
    `apply_inverse_t`+`vec_normalized` call runs exactly as it does today,
    unchanged in shape and in call count, on inputs whose VALUES (not
    necessarily computation history) are identical to today's.

    Returns ``(direction, pair_vector)`` == ``(direction_features(pos, nb),
    pair_vector_features(pos, nb, scale))``.
    """
    rot, trans, _ = aa_frames_t(pos)
    n, a, _ = pos.shape
    k = neighbours.shape[1]
    neigh_pos = pos[neighbours]                           # (N, K, A, 3)

    # `direction_features`'s OWN computation, verbatim - same shape, same
    # single call as the unmerged function, just fed a `neigh_pos` gathered
    # once instead of twice. A pure index gather has no rounding, so this is
    # bit-identical to calling `direction_features(pos, neighbours)` itself.
    direction = vec_normalized(
        apply_inverse_t(rot, trans, neigh_pos, k=2)).reshape(n, k, a * 3)

    # `pair_vector_features`'s OWN computation, verbatim - same (2A)-shaped
    # single `apply_inverse_t`/`vec_normalized` call as the unmerged function.
    # See the docstring below: concatenating self_pos/neigh_pos and reusing
    # `direction`'s already-transformed neighbour half here is NOT safe in
    # eager mode (measured, not theoretical - splitting or re-batching this
    # exact call perturbs its low bits by ~1e-7 inside a live forward pass,
    # even though every intermediate is mathematically identical). Reusing
    # only `rot`/`trans`/`neigh_pos` - none of which round anything - and
    # calling `apply_inverse_t`/`vec_normalized` exactly ONCE, at exactly the
    # shape production already uses, is the only combination verified
    # bit-identical; see RESULTS.md.
    self_pos = pos[:, None].expand(n, k, a, 3)            # (N, K, A, 3)
    both = torch.cat([self_pos, neigh_pos], dim=-2)       # (N, K, 2A, 3)
    local = apply_inverse_t(rot, trans, both, k=2)
    directions = vec_normalized(local).reshape(n, k, 2 * a * 3)
    coords = (scale * local).reshape(n, k, 2 * a * 3)
    return direction, torch.cat([directions, coords], dim=-1)
