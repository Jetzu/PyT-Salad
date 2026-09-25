"""Sparse neighbour selection.

The contract to reproduce exactly (`salad/modules/utils/geometry.py:242`):

  * a dense ``(N, N)`` distance matrix is materialised, masked entries set to
    ``+inf``; selection is a **stable argsort + head slice**, not ``topk``
  * stages *chain*: each stage receives the neighbours built so far, sets their
    distances to ``+inf`` so they cannot be picked twice, selects ``count`` more
    and concatenates. Final K is the sum of the stage counts.
  * entries whose selected distance is ``inf`` become ``-1`` ("no neighbour").
    Downstream code always masks on ``!= -1`` and never trusts the gathered row.

Tie-breaking matters here (integer sequence distances produce many ties, and
fully masked rows are all-inf), which is why every sort is ``stable=True``.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .rigid import vec_norm

__all__ = ["get_neighbours", "get_index_neighbours", "get_spatial_neighbours",
           "get_random_neighbours", "extract_neighbours", "pair_distance",
           "neighbour_mask", "gather_neighbours"]

_INF = float("inf")


def pair_distance(points: Tensor) -> Tensor:
    """Dense ``(N, N)`` Euclidean distance for ``(N, 3)`` points."""
    return vec_norm(points[:, None, :] - points[None, :, :])


def neighbour_mask(neighbours: Tensor, mask: Tensor) -> Tensor:
    """``(N, K)`` validity mask: self unmasked, neighbour unmasked, index != -1."""
    return mask[:, None] & mask[neighbours] & (neighbours != -1)


def gather_neighbours(x: Tensor, neighbours: Tensor) -> Tensor:
    """``x[neighbours]`` with -1 wrapping to the last row, as JAX/NumPy do.

    Callers must mask the result; the wrapped row is never meaningful.
    """
    return x[neighbours]


def get_neighbours(count: int):
    """Core selector. Returns ``inner(distance, mask, neighbours=None)``."""

    def inner(distance: Tensor, mask: Tensor, neighbours: Tensor | None = None) -> Tensor:
        n = distance.shape[0]
        index = torch.arange(n, device=distance.device)
        distance = torch.where(mask.bool(), distance, torch.full_like(distance, _INF))
        if neighbours is not None:
            # forbid re-selecting anything already chosen (and ignore -1 slots)
            update = torch.where(
                neighbours != -1,
                torch.full_like(distance[index[:, None], neighbours], _INF),
                distance[index[:, None], neighbours],
            )
            distance = distance.clone()
            distance[index[:, None], neighbours] = update
        if count <= 0:
            knn = torch.zeros((n, 0), dtype=torch.long, device=distance.device)
        else:
            order = torch.argsort(distance, dim=-1, stable=True)
            knn = order[..., :count]
            picked = distance[index[:, None], knn]
            knn = torch.where(picked < _INF, knn, torch.full_like(knn, -1))
        if neighbours is not None:
            knn = torch.cat((neighbours, knn), dim=-1)
        return knn

    return inner


def get_index_neighbours(count: int):
    """Nearest in *sequence* position, within the same chain and batch item."""

    def inner(resi: Tensor, chain: Tensor, batch: Tensor, mask: Tensor,
              neighbours: Tensor | None = None) -> Tensor:
        same_item = batch[:, None] == batch[None, :]
        same_chain = chain[:, None] == chain[None, :]
        pair_mask = mask[:, None] & mask[None, :] & same_item & same_chain
        distance = (resi[:, None] - resi[None, :]).abs().float()
        return get_neighbours(count)(distance, pair_mask, neighbours)

    return inner


def get_spatial_neighbours(count: int):
    """k-nearest in space. `points` is already the CA slice, ``(N, 3)``."""

    def inner(points: Tensor, batch: Tensor, mask: Tensor,
              neighbours: Tensor | None = None) -> Tensor:
        same_item = batch[:, None] == batch[None, :]
        pair_mask = mask[:, None] & mask[None, :] & same_item
        distance = pair_distance(points)
        distance = torch.where(same_item, distance, torch.full_like(distance, _INF))
        return get_neighbours(count)(distance, pair_mask, neighbours)

    return inner


def get_random_neighbours(count: int):
    """Gumbel-top-k sampling without replacement, with weight proportional to 1/d^3.

    ``logits = -3*log(d + 1e-6) + G``, ``G ~ Gumbel(0,1)``; upstream forms the
    Gumbel as ``log(-log(U))`` which is ``-G``, then subtracts it. Sorting is
    then done ascending on ``-logits``.

    `points_or_distance` is overloaded exactly as upstream: a square matrix is
    taken as a precomputed distance matrix, otherwise ``(N, 3)`` points.
    """

    def inner(points_or_distance: Tensor, batch: Tensor, mask: Tensor,
              neighbours: Tensor | None = None,
              generator: torch.Generator | None = None,
              uniform: Tensor | None = None) -> Tensor:
        x = points_or_distance
        if x.dim() == 2 and x.shape[0] == x.shape[1]:
            distance = x
        else:
            distance = pair_distance(x)
        same_item = batch[:, None] == batch[None, :]
        weight = -3.0 * torch.log(distance + 1e-6)
        if uniform is None:
            uniform = torch.empty_like(weight).uniform_(
                1e-6, 1.0 - 1e-6, generator=generator)
        else:
            # injected field, used to make the draw reproducible against JAX
            uniform = uniform.to(device=weight.device, dtype=weight.dtype)
        gumbel = torch.log(-torch.log(uniform))
        weight = weight - gumbel
        distance = -weight
        distance = torch.where(same_item, distance, torch.full_like(distance, _INF))
        pair_mask = mask[:, None] & mask[None, :] & same_item
        return get_neighbours(count)(distance, pair_mask, neighbours)

    return inner


def extract_neighbours(num_index: int = 16, num_spatial: int = 16, num_random: int = 32):
    """The composite selector used by every DiffusionBlock.

    Stages chain in this order: sequence-index, then spatial on CA, then random
    on CA. Returns ``(N, num_index + num_spatial + num_random)``.
    """

    def inner(pos: Tensor, resi: Tensor, chain: Tensor, batch: Tensor, mask: Tensor,
              generator: torch.Generator | None = None,
              uniform: Tensor | None = None) -> Tensor:
        neighbours = get_index_neighbours(num_index)(resi, chain, batch, mask)
        ca = pos[:, 1]
        neighbours = get_spatial_neighbours(num_spatial)(ca, batch, mask, neighbours)
        neighbours = get_random_neighbours(num_random)(
            ca, batch, mask, neighbours, generator=generator, uniform=uniform)
        return neighbours

    return inner
