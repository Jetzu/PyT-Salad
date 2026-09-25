"""Rigid-body geometry on plain tensors.

salad's JAX code uses `Vec3Array` / `Rot3Array` / `Rigid3Array`, which are
struct-of-arrays JAX pytrees (three separate `(...)` arrays for x/y/z). That
layout exists to dodge TPU matmul pathologies and has no benefit in PyTorch, so
here a vector is simply a `Tensor[..., 3]` and a rotation a `Tensor[..., 3, 3]`.

The one construct worth keeping is `Rigid`, because salad leans heavily on the
broadcast idiom ``frames[:, None, None].apply_inverse_to_point(points)``. Naive
indexing of a packed ``(..., 3, 4)`` tensor would hit the trailing dims, so
``Rigid.__getitem__`` inserts axes *before* them.

Convention notes carried over from the reference:
  * rotations are row-major: ``to_array()[..., i, j]`` is element (i, j)
  * a packed rigid is ``cat([R (...,3,3), t[..., None] (...,3,1)], -1)``
  * ``norm`` clamps *before* the sqrt; clamping after gives NaN gradients at 0
"""
from __future__ import annotations

import torch

from typing import NamedTuple

import torch
from torch import Tensor

__all__ = [
    "vec_norm", "vec_norm2", "vec_normalized", "vec_dot", "vec_cross",
    "rot_apply", "rot_apply_inverse", "rot_compose", "rot_from_two_vectors",
    "Rigid",
]

# salad uses eps=1e-6 on the *norm*, i.e. eps**2 on the squared norm.
_NORM_EPS = 1e-6


def vec_dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(-1)


def vec_norm2(a: Tensor) -> Tensor:
    """Squared norm. No epsilon - matches `Vec3Array.norm2`."""
    return (a * a).sum(-1)


def vec_norm(a: Tensor, eps: float = _NORM_EPS) -> Tensor:
    """Euclidean norm, clamped *inside* the sqrt as `Vec3Array.norm` does."""
    return torch.sqrt(torch.clamp(vec_norm2(a), min=eps * eps))


def vec_normalized(a: Tensor, eps: float = _NORM_EPS) -> Tensor:
    return a / vec_norm(a, eps).unsqueeze(-1)


def vec_cross(a: Tensor, b: Tensor) -> Tensor:
    return torch.cross(a, b, dim=-1)


def rot_apply(rot: Tensor, point: Tensor) -> Tensor:
    """R @ p  for R (..., 3, 3) and p (..., 3), broadcasting over leading dims."""
    return torch.einsum("...ij,...j->...i", rot, point)


def rot_apply_inverse(rot: Tensor, point: Tensor) -> Tensor:
    """R^T @ p. Rotations are orthonormal, so the inverse is the transpose."""
    return torch.einsum("...ji,...j->...i", rot, point)


def rot_compose(a: Tensor, b: Tensor) -> Tensor:
    return a @ b


def rot_from_two_vectors(e0_unnormalized: Tensor, e1_unnormalized: Tensor) -> Tensor:
    """Gram-Schmidt frame, matching `Rot3Array.from_two_vectors`.

    The first vector defines the x-axis; the second is projected into the
    xy-plane and defines y. Returns ``(..., 3, 3)`` whose *columns* are the
    basis vectors (so ``rot_apply(R, [1,0,0]) == e0``).
    """
    e0 = vec_normalized(e0_unnormalized)
    # make e1 perpendicular to e0
    c = vec_dot(e1_unnormalized, e0).unsqueeze(-1)
    e1 = vec_normalized(e1_unnormalized - c * e0)
    e2 = vec_cross(e0, e1)
    # columns are (e0, e1, e2)
    return torch.stack([e0, e1, e2], dim=-1)


class Rigid:
    """A rigid transform: rotation ``(..., 3, 3)`` + translation ``(..., 3)``.

    Deliberately not an nn.Module - these are values, never parameters.

    A NamedTuple was tried here, to stop Dynamo breaking the graph on operator
    overloads (``builtin: matmul [UserDefinedObjectVariable, ...]``). It made
    things worse: as a tuple subclass it becomes a pytree node, and inductor
    then unflattens it with ``_make`` against the wrong arity
    ("Expected 2 arguments, got 50") inside the compiled region. Reverted.
    Reducing the graph-break count is still worth doing, but not this way.
    """

    __slots__ = ("rotation", "translation")

    def __init__(self, rotation: Tensor, translation: Tensor):
        self.rotation = rotation
        self.translation = translation

    # -- construction --------------------------------------------------
    @staticmethod
    def from_array(array: Tensor) -> "Rigid":
        """Unpack a ``(..., 3, 4)`` tensor (rotation | translation column)."""
        return Rigid(array[..., :3], array[..., -1])

    def to_array(self) -> Tensor:
        return torch.cat([self.rotation, self.translation.unsqueeze(-1)], dim=-1)

    @staticmethod
    def from_reference(a: Tensor, b: Tensor, c: Tensor) -> "Rigid":
        """salad's `make_transform_from_reference(a=N, b=CA, c=C)`.

        x-axis along C-CA, N placed in the +xy half plane, origin at CA.
        """
        return Rigid(rot_from_two_vectors(c - b, a - b), b)

    # -- shape plumbing ------------------------------------------------
    @property
    def shape(self) -> torch.Size:
        return self.translation.shape[:-1]

    def __len__(self) -> int:
        return self.shape[0]

    # NOTE: `__getitem__` is deliberately NOT overridden. On a NamedTuple it is
    # tuple indexing, and Dynamo asserts on any non-int key, so an override
    # would trade one graph break for a hard compile failure. The two things
    # that indexing was used for get explicit methods instead.

    def unsqueeze_batch(self, k: int = 1) -> "Rigid":
        """Insert `k` axes *after* the batch dims and *before* the trailing ones.

        Replaces the JAX broadcast idiom ``frames[:, None, None]``: rotation
        carries two trailing dims and translation one, so a naive index would
        hit those instead of the batch axis.
        """
        rot, trans = self.rotation, self.translation
        for _ in range(k):
            rot = rot.unsqueeze(-3)
            trans = trans.unsqueeze(-2)
        return Rigid(rot, trans)

    def select_batch(self, dim: int, index: int) -> "Rigid":
        """Pick one entry along a batch dimension (e.g. a torsion group)."""
        return Rigid(self.rotation.select(dim, index),
                     self.translation.select(dim, index))

    # -- the actual transforms -----------------------------------------
    def apply_to_point(self, point: Tensor) -> Tensor:
        return rot_apply(self.rotation, point) + self.translation

    def apply_inverse_to_point(self, point: Tensor) -> Tensor:
        return rot_apply_inverse(self.rotation, point - self.translation)

    def inverse(self) -> "Rigid":
        inv_rot = self.rotation.transpose(-1, -2)
        return Rigid(inv_rot, -rot_apply(inv_rot, self.translation))

    def compose(self, other: "Rigid") -> "Rigid":
        return Rigid(rot_compose(self.rotation, other.rotation),
                     self.apply_to_point(other.translation))

    __matmul__ = compose

    def to(self, *args, **kwargs) -> "Rigid":
        return Rigid(self.rotation.to(*args, **kwargs),
                     self.translation.to(*args, **kwargs))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Rigid(shape={tuple(self.shape)}, device={self.translation.device})"
