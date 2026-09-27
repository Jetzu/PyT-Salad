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

--- N036: the dedicated symmetric-generation script's extra flags -----------

Upstream ships a SECOND, separate symmetry entry point,
`salad/training/eval_sym.py` ("sample cyclic proteins"), with its own
`Screw`/`ScrewVP` classes duplicating the geometry above plus several knobs
`eval_noise_schedule_benchmark.py` (the script §1 above is drawn from) never
had. `.agent-research/results/N036-symmetry-flags/` has the full read; the
short version, per flag:

  * ``--fixcenter_threshold`` -> `do_radius` on `screw_replicate`, below.
    Upstream's `do_radius = ts > (config.fixcenter_threshold or 40.0)` with
    `ts = data["t_pos"][0]` - the CURRENT step's *scaled* noise level (after
    `--timescale_pos`), not the raw fraction `sample()` uses for
    `--sym_threshold`. Gates two things together, in both the x0 branch
    (hook C) AND the noised-input branch (hook B, where it is genuinely
    threaded through unlike `mean`): whether the canonicalised subunit is
    re-centered on its own CA mean (`docenter`) and whether the fixed
    `--screw_radius` offset is re-applied at all. Default `0.0001` (upstream's
    own CLI default) is a no-op for every schedule this port runs end to end -
    `do_radius` stays `True` the whole trajectory.
  * ``--compact_lr`` / ``--clash_lr`` -> `compact_clash_update`, below. Two
    ADDITIVE gradient-style guidance nudges upstream's `model_step` applies to
    the x0 prediction, immediately after the canonicalise+replicate step and
    before it gets noised again - i.e. inside hook C, right after
    `screw_replicate`. `compact_lr` pulls every atom toward its OWN subunit's
    mean CA position (a per-subunit compactness pull); `clash_lr` pushes CA
    atoms of DIFFERENT chains (or the same chain >16 residues apart) apart
    below an 8 A soft threshold via `-grad` of a plain relu clash cost. This is
    a THIRD, distinct clash formula from `clash_repulsion_grad`'s smooth 6/12
    switch below (upstream itself has at least three; see that function's own
    docstring) - not a rename, a different penalty on a different pair set
    (cross-subunit here, vs the representative subunit's own self-clashes
    there). Both learning rates default `0.0`, an exact no-op.
  * ``--mix_output`` -> folded into `mean` at parse time, below.
    `eval_sym.py` has no `--sym_mean` at all; it names the exact same
    couple-vs-select choice `--mix_output` (`"couple"` -> average/Procrustes,
    anything else -> pick the middle subunit) and applies it every step to
    the SAME hook-C/A call site `--sym_mean` already drives in this port. Not
    a second mechanism - upstream's other name for this one. `''` (default,
    unset) leaves `--sym_mean` in charge unchanged; a non-empty value
    overrides it. Hook B still always resolves to `mean=False` regardless of
    either flag - that inconsistency is specific to hook B, not to how `mean`
    is sourced.
  * ``--sym_noise`` -> `SymmetrySpec.sym_noise`, gates hook B's call
    entirely (in the CLI's `step_fn`, not here). Upstream:
    `config.replicate_noise = (opt.sym_noise == "True")`; when off, each
    subunit copy is noised independently by the ordinary per-residue noise
    draw and hook B's `screw_replicate` call on `pos_noised` is skipped
    outright, not merely defaulted. Default `"True"` reproduces this port's
    prior (unconditional) hook B behaviour exactly.
  * ``--mode`` -> a hard, immediate error for `"rotation"`. See
    `parse_sym`'s docstring: upstream's `ScrewVP.replicate_pos(self, first)`
    does not accept the `do_radius` argument `model_step` always passes it,
    so every `--mode rotation` invocation of `eval_sym.py` raises `TypeError`
    inside JAX's trace on step 0. Not portable, and not worth a from-scratch
    reimplementation with no working upstream referent to match - see the
    N036 notebook for the reproduction. `"screw"` (default) is exactly what
    `--sym` already implements.
  * ``--f_small`` / ``--f_strand`` -> a CLI-level reject-and-retry filter
    (`scripts/generate.py::_sym_filter_check`), not model-side geometry:
    upstream's per-out-step ALA/GLY and DSSP-strand fraction checks, which
    discard and regenerate a whole design attempt (no partial credit for
    passing some but not all requested out-steps) if either is exceeded.
    Both default `1.0`, an exact no-op (a fraction is never `> 1.0`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["SymmetrySpec", "parse_sym", "rot_y", "cyclic_centering",
          "screw_replicate", "clash_repulsion_grad", "compact_clash_update"]


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
    #: N036 additions - all upstream defaults, all no-ops until set.
    sym_noise: bool = True
    fixcenter_threshold: float = 0.0001
    compact_lr: float = 0.0
    clash_lr: float = 0.0
    f_small: float = 1.0
    f_strand: float = 1.0
    mix_output: str = ""


def parse_sym(sym: str, *, replicate_count: int | None, screw_angle: float | None,
             screw_radius: float, screw_translation: float, sym_mean: str,
             sym_threshold: float, mode: str = "screw", mix_output: str = "",
             sym_noise: str = "True", fixcenter_threshold: float = 0.0001,
             compact_lr: float = 0.0, clash_lr: float = 0.0,
             f_small: float = 1.0, f_strand: float = 1.0) -> SymmetrySpec:
    """``--sym`` (+ friends) -> `SymmetrySpec`.

    ``sym_mean`` is compared as a literal string (``"True"``/``"False"``), not
    parsed as a bool, matching the spec's literal ``mean=(sym_mean == "True")``
    - reproduced here rather than "improved" to `argparse`'s `store_true` so a
    command line copied from notes/specs still does the right thing.
    ``sym_noise`` follows the same convention (upstream's
    `opt.sym_noise == "True"`).

    ``"cN"`` (e.g. ``"c4"``) is a convenience shorthand for pure cyclic (no
    screw rise) symmetry: it sets `replicate_count=N`, `angle=360/N deg`,
    `translation=0` as *defaults*, overridden by any of `--replicate_count`
    /`--screw_angle`/`--screw_translation` the caller passed explicitly.

    ``mode="rotation"`` raises immediately rather than being ported: upstream
    `salad/training/eval_sym.py` selects this via
    ``sym_op_type = ScrewVP if opt.mode == "rotation" else Screw`` and then
    unconditionally calls
    ``sym_op.replicate_pos(mix_output(pos, do_radius), do_radius)`` every
    step. `ScrewVP.replicate_pos` is defined as ``def replicate_pos(self,
    first):`` - it does not declare (or use) a `do_radius` parameter at all,
    unlike its own `couple_pos`/`select_pos`, which accept but ignore it. That
    second, extra positional argument raises ``TypeError: replicate_pos()
    takes 2 positional arguments but 3 were given`` the first time `step` is
    traced - i.e. `--mode rotation` has never produced a single design in any
    released version of `eval_sym.py`. Reproduced/verified standalone (no jax
    needed - a pure Python arg-count mismatch) in
    `.agent-research/results/N036-symmetry-flags/`. `mix_output` here is
    folded into `mean`, exactly mirroring upstream's own
    ``mix_output = couple_pos if opt.mix_output == "couple" else select_pos``
    dispatch (`"couple"` -> average/Procrustes, i.e. `mean=True`; anything
    else -> pick-the-middle-subunit, `mean=False`) - it overrides `sym_mean`
    when non-empty, otherwise `sym_mean` is unchanged from before this flag
    existed.
    """
    sym = (sym or "").strip()
    if not sym:
        return SymmetrySpec(active=False)

    if mode not in ("screw", "rotation"):
        raise ValueError(f"--mode must be 'screw' or 'rotation', got {mode!r}")
    if mode == "rotation":
        raise SystemExit(
            "--mode rotation is not portable. Upstream's "
            "salad/training/eval_sym.py::ScrewVP.replicate_pos(self, first) "
            "does not accept the `do_radius` argument that model_step "
            "always passes at the call site "
            "(`sym_op.replicate_pos(mix_output(pos, do_radius), "
            "do_radius)`), so every --mode rotation invocation of that "
            "script raises TypeError inside JAX's trace on step 0, in every "
            "released version - this mode has never produced a design "
            "upstream. See .agent-research/results/N036-symmetry-flags/ "
            "for the reproduction (a pure Python arg-count check, no jax "
            "needed). Only 'screw' (the default, and the only mode this "
            "port implements) is portable.")

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

    mean = (sym_mean == "True")
    if mix_output:
        mean = (mix_output == "couple")

    return SymmetrySpec(active=True, group=sym, replicate_count=count,
                        angle_rad=angle_deg / 180.0 * math.pi,
                        radius=screw_radius, translation=translation,
                        mean=mean, threshold=sym_threshold,
                        sym_noise=(sym_noise == "True"),
                        fixcenter_threshold=fixcenter_threshold,
                        compact_lr=compact_lr, clash_lr=clash_lr,
                        f_small=f_small, f_strand=f_strand,
                        mix_output=mix_output)


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
                    time: float = 0.0, do_radius: bool = True) -> Tensor:
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

    ``do_radius`` (N036, default `True` - every pre-existing call site keeps
    today's behaviour unchanged): upstream `eval_sym.py`'s `do_radius` gate,
    ``--fixcenter_threshold`` in the CLI. When `False`, the canonicalisation
    step does NOT re-center the representative subunit on its own CA mean
    (``docenter=False``) and the fixed `radius` offset is not re-applied at
    all - not "applied at 0", genuinely skipped, matching upstream's
    ``first.at[:].add([do_radius * radius, 0, 0])`` (0 contribution either
    way, so the two are output-identical, but this reads closer to the
    source).
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
            centered = cyclic_centering(seg, docenter=do_radius)
            acc = acc + centered @ rot_pow[i].transpose(-1, -2)
        first = acc / count
    else:
        i = count // 2
        seg = used[i * size:(i + 1) * size]
        first = cyclic_centering(seg, docenter=do_radius) @ rot_pow[i].transpose(-1, -2)

    if chain_center:
        ca = first[:, 1, :]
        grad = clash_repulsion_grad(ca)
        first = first + (time * 5.0 * grad)[:, None, :]

    if do_radius:
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


def compact_clash_update(pos: Tensor, *, replicate_count: int, chain_index: Tensor,
                         residue_index: Tensor, compact_lr: float, clash_lr: float,
                         t: float) -> Tensor:
    """``--compact_lr`` / ``--clash_lr``. `salad/training/eval_sym.py:180-213`.

    Two additive, gradient-style guidance nudges on the x0 prediction, meant
    to run in hook C right after `screw_replicate` (i.e. on the just
    canonicalised/replicated `pos`, before it is noised again for the next
    step) - upstream computes both from the SAME pre-update `ca`/`pos`
    (neither term sees the other's delta) and adds both, scaled by
    `compact_lr`/`clash_lr` and the current step's *scaled* noise level `t`
    (upstream reads this off ``data["t_pos"]``, not the raw step fraction).

    ``compact_lr``: pulls every atom of each residue toward its OWN
    subunit's mean CA position (`ca.reshape(count, L//count, 3).mean(axis=1)`,
    broadcast back) - a per-subunit compactness pull, not an inter-subunit
    one.

    ``clash_lr``: `-grad` of a plain relu clash cost
    (``relu(8 - dist) / 8``, summed) between CA pairs that are either in a
    DIFFERENT chain, or in the same chain more than 16 residues apart in
    `residue_index` - i.e. it never penalises local (same-chain, <=16 apart)
    geometry, only cross-subunit / long-range contacts. This is a THIRD,
    distinct clash formula from `clash_repulsion_grad` above (a plain relu
    penalty on cross-subunit pairs here, vs a smooth 6/12-switch penalty on
    one subunit's own self-clashes there) - upstream itself has at least
    three unrelated "clash" formulas across its eval scripts; see that
    function's docstring.

    `compact_lr == clash_lr == 0.0` (the CLI defaults) is an exact no-op -
    the function returns `pos` unchanged without even building the pair
    masks.
    """
    if compact_lr == 0.0 and clash_lr == 0.0:
        return pos
    L = pos.shape[0]
    num_rep = max(replicate_count, 1)
    if num_rep <= 1 or L % num_rep != 0:
        return pos
    per_rep = L // num_rep
    device, dtype = pos.device, pos.dtype
    ca = pos[:, 1, :]
    delta = torch.zeros_like(pos)

    if compact_lr != 0.0:
        mean_pos = ca.reshape(num_rep, per_rep, 3).mean(dim=1, keepdim=True)
        mean_pos = mean_pos.expand(num_rep, per_rep, 3).reshape(L, 3)
        compact_update = mean_pos[:, None, :] - pos
        delta = delta + compact_lr * compact_update

    if clash_lr != 0.0:
        chain_index = chain_index.to(device=device)
        residue_index = residue_index.to(device=device)
        same_chain = chain_index[:, None] == chain_index[None, :]
        inf = torch.full((L, L), float("inf"), device=device, dtype=dtype)
        rdist = torch.where(
            same_chain,
            (residue_index[:, None] - residue_index[None, :]).abs().to(dtype),
            inf)
        long_range = rdist > 16
        with torch.enable_grad():
            leaf = ca.detach().clone().requires_grad_(True)
            dist = torch.sqrt(torch.clamp(
                ((leaf[:, None, :] - leaf[None, :, :]) ** 2).sum(-1), min=1e-6))
            clashyness = torch.where(
                long_range, torch.relu(8.0 - dist) / 8.0,
                torch.zeros_like(dist)).sum()
            grad, = torch.autograd.grad(clashyness, leaf)
        clash_update = -grad.detach()
        delta = delta + clash_lr * clash_update[:, None, :]

    return pos + t * delta
