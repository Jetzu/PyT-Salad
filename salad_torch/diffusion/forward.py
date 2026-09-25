"""Forward (noising) process: VE, VP, flow, and sequence masking.

Port of `salad.modules.utils.diffusion`. Every noiser here is split into a
*pure* transform and a *sampler*, and the wrapper takes an optional pre-drawn
noise tensor. That split is not cosmetic - it is the only way to test this port:
JAX's PRNG is not reproducible from torch, so parity is checked by dumping
`jax.random.normal` output and replaying it through the pure path. If you add a
noiser, keep the split.

Three injection levels are available, from highest to lowest:

* ``noise=`` - the finished ``(N, A, 3)`` perturbation *before* centring, i.e.
  already multiplied by ``cloud_std`` / ``residue_std``.
* ``noise_std_normal=`` - the raw standard normal, before any scaling. This is
  the level at which `jax.random.normal` output can be dropped in verbatim.
* ``generator=`` - a `torch.Generator`, for reproducible torch-only runs.

Note that `mask` never suppresses noise anywhere in this module: masked
residues are noised like any other, and `mask` is consumed only by
`position_std`. That is upstream behaviour, not an oversight.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from ..geometry.features import fourier_time_embedding
from ..geometry.segment import index_mean

__all__ = [
    "position_std", "center_noise_on_ca", "apply_atom_cloud_noise",
    "sample_atom_cloud_noise", "diffuse_atom_cloud", "diffuse_coordinates_edm",
    "diffuse_sequence", "fourier_time_embedding",
]

SymmFn = Callable[[Tensor], Tensor]


def _broadcast_t(t: float | Tensor, num_residues: int,
                 like: Tensor) -> Tensor | float:
    """Reshape a time/sigma argument so it broadcasts over ``(N, A, 3)``.

    Upstream passes ``t_pos[:, None]``, which is right for `Vec3Array`'s
    ``(N, A)``-per-component layout but one axis short for a dense
    ``(N, A, 3)`` tensor - hence ``(N, 1, 1)`` here.
    """
    if not torch.is_tensor(t):
        return float(t)
    if t.dim() == 0:
        return t
    t = t.reshape(num_residues, *([1] * (like.dim() - 1)))
    return t.to(device=like.device, dtype=like.dtype)


def position_std(x: Tensor, index: Tensor, mask: Tensor,
                 eps: float = 0.0) -> Tensor:
    """Per-segment spatial spread of the CA atoms. ``diffusion.py:52-67``.

    Returns ``(N,)``: per residue, but constant within each segment. The three
    Cartesian variances are averaged *before* the square root, so this is
    ``sqrt(mean_xyz(var))`` (roughly ``Rg / sqrt(3)``), not the norm-based std.

    `eps` defaults to 0.0, which is bit-identical to upstream - and upstream has
    no floor at all, so a single-residue or fully-masked segment can leave a
    tiny negative under the sqrt and return NaN. Raise `eps` when debugging;
    leave it at 0.0 for parity.
    """
    # x[:, 1] is a view in torch; upstream's `ca -= center` would mutate the
    # caller's coordinates here, so subtract out of place.
    ca = x[:, 1] - index_mean(x[:, 1], index, mask[..., None])
    second = index_mean(ca ** 2, index, mask[..., None]).mean(dim=1)
    first = index_mean(ca, index, mask[..., None]).mean(dim=1) ** 2
    return torch.sqrt(second - first + eps)


def center_noise_on_ca(target: Tensor, batch_index: Tensor) -> Tensor:
    """Subtract each batch item's *CA-noise* centroid from all of its atoms.

    The centre comes from atom 1 only but is removed from all A atoms, and the
    divisor is the raw residue count - `mask` is not consulted. Both are
    upstream behaviour (``diffusion.py:34-40``, ``:93-97``).
    """
    ones = torch.ones(target.shape[0], 1, dtype=torch.bool, device=target.device)
    center = index_mean(target[:, 1], batch_index, ones)
    return target - center[:, None, :]


def apply_atom_cloud_noise(x: Tensor, target: Tensor, t: float | Tensor,
                           *, batch_index: Tensor, flow: bool = False,
                           symm: SymmFn | None = None) -> Tensor:
    """Pure VP/flow mix. `target` must already carry the ``cloud_std`` scaling.

    VP uses ``x * sqrt(1 - t**2) + target * t``, i.e. ``(alpha, sigma) =
    (sqrt(1 - sigma^2), sigma)`` - consistent with the cosine schedule's
    ``alpha_bar = 1 - sigma^2``. Flow uses the linear ``x * (1 - t) + target * t``.
    """
    target = center_noise_on_ca(target, batch_index)
    if symm is not None:
        target = symm(target)
    t_t = torch.as_tensor(_broadcast_t(t, x.shape[0], x), dtype=x.dtype,
                          device=x.device)
    if flow:
        return x * (1 - t_t) + target * t_t
    return x * torch.sqrt(1 - t_t ** 2) + target * t_t


def sample_atom_cloud_noise(shape: tuple[int, int, int],
                            cloud_std: float | Tensor,
                            *, residue_std: float = 4.0,
                            correlated_residues: bool = False,
                            generator: torch.Generator | None = None,
                            device: torch.device | str | None = None,
                            dtype: torch.dtype = torch.float32,
                            std_normal: Tensor | tuple[Tensor, Tensor] | None = None
                            ) -> Tensor:
    """Draw the ``(N, A, 3)`` perturbation, before centring.

    The `correlated_residues` branch is the only place the atom cloud consumes
    *two* RNG draws (cloud first, then per-atom jitter) and the only place
    `residue_std` is used - unscaled by `cloud_std`. No shipped config or call
    site sets it; it exists for completeness.
    """
    n, a, _ = shape
    if correlated_residues:
        if std_normal is None:
            cloud = torch.randn((n, 1, 3), generator=generator, device=device,
                                dtype=dtype)
            residue = torch.randn((n, a, 3), generator=generator, device=device,
                                  dtype=dtype)
        else:
            cloud, residue = std_normal
        return cloud_std * cloud + residue_std * residue
    if std_normal is None:
        std_normal = torch.randn(shape, generator=generator, device=device,
                                 dtype=dtype)
    return cloud_std * std_normal


def diffuse_atom_cloud(x: Tensor, mask: Tensor, batch_index: Tensor,
                       t: float | Tensor, cloud_std: float | Tensor | None = None,
                       flow: bool = False, residue_std: float = 4.0,
                       correlated_residues: bool = False,
                       symm: SymmFn | None = None, *,
                       noise: Tensor | None = None,
                       noise_std_normal: Tensor | tuple[Tensor, Tensor] | None = None,
                       generator: torch.Generator | None = None,
                       return_noise: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    """VP / VP-scaled / flow noising. ``diffusion.py:69-114``.

    `cloud_std` is either a scalar (10.0 = ``c.sigma_data`` for ``vpfixed`` and
    ``flow``, or the user's ``--cloud_std``) or `None`, in which case it is
    derived per segment from `position_std` and broadcast as ``(N, 1, 1)`` so
    every atom of a residue is scaled by that residue's segment spread.

    `noise` is taken verbatim as the already-scaled, un-centred perturbation;
    `noise_std_normal` is the unscaled standard normal. Either suppresses all
    RNG use.
    """
    if cloud_std is None:
        cloud_std = position_std(x, batch_index, mask)[:, None, None]
    if noise is None:
        noise = sample_atom_cloud_noise(
            tuple(x.shape), cloud_std, residue_std=residue_std,
            correlated_residues=correlated_residues, generator=generator,
            device=x.device, dtype=x.dtype, std_normal=noise_std_normal)
    result = apply_atom_cloud_noise(x, noise, t, batch_index=batch_index,
                                    flow=flow, symm=symm)
    return (result, noise) if return_noise else result


def diffuse_coordinates_edm(x: Tensor, batch_index: Tensor, t: float | Tensor,
                            symm: SymmFn | None = None, *,
                            noise: Tensor | None = None,
                            generator: torch.Generator | None = None,
                            return_noise: bool = False
                            ) -> Tensor | tuple[Tensor, Tensor]:
    """VE noising: ``x + t * centred_normal``. ``diffusion.py:27-50``.

    Only one draw, and `mask` plays no part - not even in the centroid divisor.
    """
    if noise is None:
        noise = torch.randn(x.shape, generator=generator, device=x.device,
                            dtype=x.dtype)
    target = center_noise_on_ca(noise, batch_index)
    if symm is not None:
        target = symm(target)
    result = x + _broadcast_t(t, x.shape[0], x) * target
    return (result, noise) if return_noise else result


def diffuse_sequence(x: Tensor, t: float | Tensor, mask_index: int = 20,
                     *, uniform: Tensor | None = None,
                     generator: torch.Generator | None = None
                     ) -> tuple[Tensor, Tensor]:
    """Mask each position to `mask_index` with per-residue probability `t`.

    ``diffusion.py:12-25``. JAX's `bernoulli` is ``uniform(key, shape) < p``, so
    the comparison is strictly less-than; injecting the raw uniform rather than
    the boolean keeps that comparison inside the port where it can be tested.

    At inference with ``default_vp``, ``t_seq`` is 1.0 everywhere so every
    position is masked - but the draw still happens and still advances the
    stream, which matters if you ever chase RNG-level parity.
    """
    if uniform is None:
        uniform = torch.rand(x.shape, generator=generator, device=x.device,
                             dtype=torch.float32)
    t_t = t if torch.is_tensor(t) else torch.as_tensor(
        t, dtype=uniform.dtype, device=uniform.device)
    corrupt_aa = uniform < t_t
    out = torch.where(corrupt_aa,
                      torch.full_like(x, mask_index), x)
    return out, corrupt_aa
