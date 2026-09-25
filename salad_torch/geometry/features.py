"""Distance binning, relative-position encodings and pair features."""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .rigid import Rigid, vec_norm, vec_normalized
from .frames import extract_aa_frames
from .segment import index_count

__all__ = [
    "linear_centers", "gaussian_rbf", "distance_rbf", "distance_one_hot",
    "fourier_time_embedding", "sequence_relative_position",
    "distance_features", "direction_features", "rotation_features",
    "position_rotation_features", "pair_vector_features",
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
                               cyclic: bool = False, identify_ends: bool = False):
    """Relative residue-index encoding.

    Bins: ``clip(resi_i - resi_j, -count, count) + count`` in ``[0, 2*count]``,
    with cross-chain / cross-batch pairs placed in the extra bin ``2*count+1``.
    ``count_total = 2*count + 2`` (= 66 for count=32).
    """
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
    frames, _ = extract_aa_frames(pos)
    # neighbour atom positions expressed in frame i
    neighbour_pos = pos[neighbours]                       # (N, K, A, 3)
    local = frames[:, None, None].apply_inverse_to_point(neighbour_pos)
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
    frames, _ = extract_aa_frames(pos)
    return rotation_features(frames, neighbours)


def pair_vector_features(pos: Tensor, neighbours: Tensor, scale: float = 0.1) -> Tensor:
    """Directions and scaled coordinates of both residues' atoms in frame i.

    -> ``(N, K, 12*A)``: ``[dir(2A*3), scale*coord(2A*3)]``.
    """
    frames, _ = extract_aa_frames(pos)
    n, a, _ = pos.shape
    k = neighbours.shape[1]
    self_pos = pos[:, None].expand(n, k, a, 3)            # (N, K, A, 3)
    neigh_pos = pos[neighbours]                           # (N, K, A, 3)
    both = torch.cat([self_pos, neigh_pos], dim=-2)       # (N, K, 2A, 3)
    local = frames[:, None, None].apply_inverse_to_point(both)
    directions = vec_normalized(local).reshape(n, k, 2 * a * 3)
    coords = (scale * local).reshape(n, k, 2 * a * 3)
    return torch.cat([directions, coords], dim=-1)
