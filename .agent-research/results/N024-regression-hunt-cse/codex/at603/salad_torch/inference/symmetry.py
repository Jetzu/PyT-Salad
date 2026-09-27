"""``--sym`` screw/cyclic symmetrisation. `notes/specs/06_generation_flags.md` §1.

There are exactly two live call sites (see the package docstring for the third,
dead one), both built on the same `screw_replicate` primitive with different
fixed arguments:

  * Hook A - `init_pos`, once, before the design loop (`chain_center=False`).
  * Hook B - the noised input `pos_noised`, every network call
    (`chain_center=False, mean=False`, **hardcoded** regardless of
    ``--sym_mean`` - an upstream inconsistency, reproduced deliberately).
  * Hook C - the x0 prediction, every step, gated by ``--sym_threshold``
    (`chain_center=(raw_t > 0.5), time=raw_t`).

None of these need a new hook in `sampling.py`: A and B are plain call-site
composition in the CLI, C is exactly `sample()`'s existing `post_step_fn`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["SymmetrySpec", "parse_sym", "rot_y", "cyclic_centering",
          "screw_replicate", "clash_repulsion_grad"]


@dataclass
class SymmetrySpec:
    """Parsed ``--sym`` + friends. `active=False` makes every hook a no-op.

    ``group`` carries the raw ``--sym`` string through for logging; the actual
    geometry is driven by `replicate_count` / `angle_rad` / `radius` /
    `translation` alone. The ``"cN"`` convenience parsing (see `parse_sym`) is
    a port-side addition - the spec only documents `--sym` as a boolean gate
    and leaves its string grammar to the CLI, so this keeps the common case
    (``--sym c4``) ergonomic while still letting `--replicate_count` /
    `--screw_angle` / `--screw_translation` override it explicitly.
    """

    active: bool
    group: str = ""
    replicate_count: int = 1
    angle_rad: float = 0.0
    radius: float = 0.0
    translation: float = 0.0
    mean: bool = False
    threshold: float = 0.0


def parse_sym(sym: str, *, replicate_count: int | None, screw_angle: float | None,
             screw_radius: float, screw_translation: float, sym_mean: str,
             sym_threshold: float) -> SymmetrySpec:
    """``--sym`` (+ friends) -> `SymmetrySpec`.

    ``sym_mean`` is compared as a literal string (``"True"``/``"False"``), not
    parsed as a bool, matching the spec's literal ``mean=(sym_mean == "True")``
    - reproduced here rather than "improved" to `argparse`'s `store_true` so a
    command line copied from notes/specs still does the right thing.

    ``"cN"`` (e.g. ``"c4"``) is a convenience shorthand for pure cyclic (no
    screw rise) symmetry: it sets `replicate_count=N`, `angle=360/N deg`,
    `translation=0` as *defaults*, overridden by any of `--replicate_count`
    /`--screw_angle`/`--screw_translation` the caller passed explicitly.
    """
    sym = (sym or "").strip()
    if not sym:
        return SymmetrySpec(active=False)

    count, angle_deg, translation = replicate_count, screw_angle, screw_translation
    if sym[0] in "cC" and sym[1:].isdigit():
        n = int(sym[1:])
        if count is None:
            count = n
        if angle_deg is None:
            angle_deg = 360.0 / n
        # translation left at whatever --screw_translation gave (default 0.0)
    if count is None:
        count = 1
    if angle_deg is None:
        angle_deg = 0.0

    return SymmetrySpec(active=True, group=sym, replicate_count=count,
                        angle_rad=angle_deg / 180.0 * math.pi,
                        radius=screw_radius, translation=translation,
                        mean=(sym_mean == "True"), threshold=sym_threshold)


def rot_y(angle: float, *, device=None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Rotation about Y, `angle` in **radians**. `notes/specs/...` §1.3."""
    a = torch.as_tensor(angle, device=device, dtype=dtype)
    c, s = torch.cos(a), torch.sin(a)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([
        torch.stack([c, zero, -s]),
        torch.stack([zero, one, zero]),
        torch.stack([s, zero, c]),
    ])


def cyclic_centering(x: Tensor, docenter: bool = True) -> Tensor:
    """Subtract the CA (atom index 1) mean over the leading (residue) axis."""
    if not docenter:
        return x
    return x - x[..., 1, :].mean(dim=0)


def clash_repulsion_grad(ca: Tensor) -> Tensor:
    """Gradient of the smooth CA-CA self-clash cost w.r.t. `ca`. §1.3, §1.3.1.

    Autograd, not a closed form, per the spec. Locally re-enables grad inside
    the ambient `torch.no_grad()` sampling loop and detaches the result -
    `screw_replicate` must not become part of any outer graph.
    """
    with torch.enable_grad():
        leaf = ca.detach().clone().requires_grad_(True)
        # Clamp the signed DIFFERENCE, then square. Upstream:
        #   eval_noise_schedule_benchmark.py:205
        #   dist = sqrt((maximum(x[:,None] - x[None,:], 1e-3) ** 2).sum(-1))
        # i.e. max(dx, 1e-3)**2, which zeroes negative components. Clamping
        # dx**2 instead keeps them and is a different function.
        # NB an earlier pass reverted this after checking inference/contacts.py,
        # which clamps max(sum(dx**2), 1e-6) - a THIRD formula, and not the one
        # this hook mirrors (the 6/12 switch below matches the eval script).
        diff = torch.clamp(leaf[:, None, :] - leaf[None, :, :], min=1e-3)
        d = torch.sqrt((diff ** 2).sum(-1))
        ratio = (d - 2) / 8
        switch = (torch.clamp(1 - ratio ** 6, min=1e-6)
                 / torch.clamp(1 - ratio ** 12, min=1e-6))
        loss = switch.mean()
        grad, = torch.autograd.grad(loss, leaf)
    return torch.nan_to_num(grad, nan=0.0).detach()


def screw_replicate(x: Tensor, *, count: int, angle: float, radius: float,
                    translation: float, chain_center: bool, mean: bool = False,
                    time: float = 0.0) -> Tensor:
    """Screw/cyclic-symmetrise `x` (`(L, A, 3)`) into `count` subunits. §1.3.

    ``translation != 0`` gives true screw symmetry; ``translation == 0``
    degenerates to plain C_n point-group symmetry.

    Input contract: `count` should evenly divide `L` (`--num_aa` should be
    given as `count` colon-separated, ideally equal-length, chains). If it
    doesn't, ``size = L // count`` (floor division) and the trailing
    ``L - size*count`` residues are passed through **unchanged** - not
    touched by the canonicalise/replicate math, matching the spec's "silent
    no-op on the remainder, not an error". (The spec's own pseudocode returns
    exactly `count*size` residues for the evenly-divisible case, which is the
    intended use; the passthrough here is this port's choice for how to keep
    `sample()`'s per-step ``pos = post_step_fn(pos, raw_t)`` shape-stable when
    it isn't.)
    """
    L = x.shape[0]
    size = L // max(count, 1)
    if size == 0 or count <= 1:
        return x
    device, dtype = x.device, x.dtype

    rot = rot_y(angle, device=device, dtype=dtype)
    rot_pow = [torch.linalg.matrix_power(rot, i) for i in range(count)]

    used = x[: size * count]
    remainder = x[size * count:]

    if mean:
        acc = torch.zeros((size, x.shape[1], 3), device=device, dtype=dtype)
        for i in range(count):
            seg = used[i * size:(i + 1) * size]
            centered = cyclic_centering(seg, docenter=True)
            acc = acc + centered @ rot_pow[i].transpose(-1, -2)
        first = acc / count
    else:
        i = count // 2
        seg = used[i * size:(i + 1) * size]
        first = cyclic_centering(seg, docenter=True) @ rot_pow[i].transpose(-1, -2)

    if chain_center:
        ca = first[:, 1, :]
        grad = clash_repulsion_grad(ca)
        first = first + (time * 5.0 * grad)[:, None, :]

    radial_shift = torch.tensor([radius, 0.0, 0.0], device=device, dtype=dtype)
    first = first + radial_shift

    replicates = []
    for i in range(count):
        rise = torch.tensor([0.0, i * translation, 0.0], device=device, dtype=dtype)
        replicates.append(first @ rot_pow[i] + rise)
    result = torch.cat(replicates, dim=0)
    result = result - result[:, 1, :].mean(dim=0)

    if remainder.numel():
        result = torch.cat([result, remainder], dim=0)
    return result
