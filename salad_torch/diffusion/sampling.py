"""The sampling loop.

Ported from `eval_noise_schedule_benchmark.py:592-673`, which is what the README
documents and what produced the published numbers - *not* from
`salad/inference/sampling.py`, a later refactor whose `Sampler.generate`
references an undefined `key` and raises `NameError` on first call. The few
genuine improvements in that refactor (a `start_steps` offset, copying the data
dict instead of mutating the caller's, carrying the raw update into each
snapshot) are folded in here because they are behaviour-preserving supersets.

There is **no integrator**. Each step sets a sigma, lets the model re-noise the
previous x0 prediction internally, predicts a fresh x0, and feeds it back. It is
repeated single-shot denoising - not DDPM, DDIM or an ODE solver - so there is
no step-size or ancestral-noise term to get wrong.

Three upstream bugs are reproduced deliberately and are flagged at their sites:
the ``raw_t`` off-by-one, the positional (not maximal) ``out_steps[-1]``
termination, and the self-conditioning reset comparing ``raw_t`` rather than the
scaled sigma. Together the first and third mean the default run stops at
sigma 0.32 and runs 80% of its trajectory without self-conditioning. Changing
any of them changes the model's output distribution, so they are defaults, not
accidents to be tidied away.

I/O is not done here: `sample` returns snapshots and `salad_torch.io.pdb` writes
them.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from .schedules import parse_timescale

__all__ = ["SamplerConfig", "sample", "sample_group", "make_initial_data"]

#: ``step_fn(data, prev, generator) -> (update, prev)``.
StepFn = Callable[[dict, dict, torch.Generator | None], tuple[dict, dict]]

#: Keys copied out of `data` into every snapshot, plus `result`/`aa_logits`.
_SNAPSHOT_KEYS = ("pos", "seq", "atom_pos", "aatype", "residue_index",
                  "chain_index", "batch_index", "mask", "cyclic_mask",
                  "t_pos", "t_seq", "latent")


@dataclass
class SamplerConfig:
    """Defaults are the CLI defaults of the eval script (`:455-460`)."""

    num_steps: int = 500
    out_steps: Sequence[int] = (400,)
    start_steps: int = 0
    prev_threshold: float = 0.8
    timescale_pos: str | Callable = "cosine(t)"
    timescale_seq: str | Callable | None = "1"
    #: Override the ``raw_t`` divisor. `None` keeps upstream's ``num_steps``,
    #: i.e. the off-by-one; ``num_steps - 1`` is the fixed variant the author
    #: left commented out at ``eval_...:596``. Not a default for a reason.
    raw_t_denominator: int | None = None
    #: Ask `step_fn` for the sequence heads only on steps that report (P2).
    #:
    #: Off by default. When on, `step_fn` is called with an extra
    #: ``need_aa=<bool>`` keyword and may return an update WITHOUT ``aa``,
    #: ``aatype`` and ``atom_pos``; the loop then leaves `data`'s copies
    #: alone. That is output-preserving **only** because the denoiser reads
    #: none of them back: its inputs are
    #: ``(pos_noised, prev_pos, prev_local, condition, aa_gt)``, `aa_gt` is a
    #: base key fixed for the whole run, and `data["seq"]` reaches the model
    #: solely through `apply_diffusion`'s `diffuse_sequence`, whose output
    #: `seq_noised` is consumed only when ``config.diffuse_sequence`` is set
    #: (no released checkpoint sets it). Turn this on only after checking
    #: that, and only with a `step_fn` that accepts `need_aa`.
    lean_steps: bool = False


def make_initial_data(num_aa: int, residue_index: Tensor, chain_index: Tensor,
                      cyclic_mask: Tensor | None = None, *,
                      atom_count: int = 20, local_size: int = 128,
                      cloud_std: float | None = None,
                      device: torch.device | str | None = None
                      ) -> tuple[dict, dict]:
    """Build the per-design ``(data, init_prev)`` pair. ``eval_...:531-590``.

    Everything starts at zero/MASK: positions are the origin, the sequence is
    all token 20, and both clocks are 1.0 (they are overwritten on the first
    step anyway). `atom_count` is ``5 + augment_size`` = 20 for ``default_vp``.
    """
    i32 = dict(dtype=torch.long, device=device)
    f32 = dict(dtype=torch.float32, device=device)
    init_pos = torch.zeros((num_aa, atom_count, 3), **f32)
    init_aa_gt = torch.full((num_aa,), 20, **i32)
    data: dict[str, Any] = {
        "pos": init_pos,
        "aa_gt": init_aa_gt,
        "seq": init_aa_gt,
        "residue_index": residue_index.to(device=device, dtype=torch.long),
        "chain_index": chain_index.to(device=device, dtype=torch.long),
        "batch_index": torch.zeros_like(chain_index).to(**i32),
        "mask": torch.ones((num_aa,), dtype=torch.bool, device=device),
        "t_pos": torch.ones((num_aa,), **f32),
        "t_seq": torch.ones((num_aa,), **f32),
    }
    if cyclic_mask is not None:
        data["cyclic_mask"] = cyclic_mask.to(device=device, dtype=torch.bool)
    if cloud_std is not None:
        data["cloud_std"] = cloud_std
    init_prev = {"pos": torch.zeros_like(init_pos),
                 "local": torch.zeros((num_aa, local_size), **f32)}
    return data, init_prev


def sample(step_fn: StepFn, data: dict, prev: dict, cfg: SamplerConfig, *,
           generator: torch.Generator | None = None,
           post_step_fn: Callable[[Tensor, float], Tensor] | None = None,
           final_step_fn: StepFn | None = None) -> list[dict]:
    """Run the denoising loop and return one snapshot per requested step.

    `prev` doubles as the zeroed state the self-conditioning reset falls back
    to, exactly as upstream's single `init_prev` dict does.

    `post_step_fn(pos, raw_t) -> pos` is the hook for the ``--sym`` screw
    symmetrisation (``eval_...:611-619``); `final_step_fn` is the ``--sample_final``
    second model, invoked *instead of* `step_fn` on output steps.
    """
    out_steps = list(cfg.out_steps)
    if out_steps and out_steps != sorted(out_steps):
        warnings.warn(
            f"out_steps {out_steps} is not ascending; upstream breaks on the "
            f"last element as written ({out_steps[-1]}), so later steps are "
            f"never reached. Reproducing that.", stacklevel=2)

    timescale_pos = parse_timescale(cfg.timescale_pos)
    timescale_seq = (None if cfg.timescale_seq is None
                     else parse_timescale(cfg.timescale_seq))

    # upstream mutates the caller's dict; a shallow copy is identical and safer
    data = dict(data)
    init_prev = prev
    denominator = cfg.raw_t_denominator or cfg.num_steps

    results: list[dict] = []
    for step in range(cfg.start_steps, cfg.num_steps):
        # `1 - step / num_steps`, not `/(num_steps - 1)`: FIXME at eval_...:594.
        # With the defaults this stops at raw_t = 0.2 (sigma 0.32) and never
        # reaches 0, so the written structure is an x0 prediction made from a
        # still-noisy state. Python float division, to match bit for bit.
        raw_t = 1 - step / denominator

        scaled_t_pos = timescale_pos(raw_t)
        scaled_t_seq = 1.0 if timescale_seq is None else timescale_seq(raw_t)
        data["t_pos"] = scaled_t_pos * torch.ones_like(data["t_pos"])
        data["t_seq"] = scaled_t_seq * torch.ones_like(data["t_seq"])

        # compares raw_t, not the scaled sigma (FIXME at eval_...:601). At the
        # defaults this zeroes `prev` from step 101 on, so self-conditioning is
        # live for only the first 20% of the trajectory.
        if raw_t < cfg.prev_threshold:
            prev = init_prev

        is_out_step = step in out_steps
        fn = final_step_fn if (final_step_fn is not None and is_out_step) else step_fn
        if cfg.lean_steps:
            update, prev = fn(data, prev, generator, need_aa=is_out_step)
        else:
            update, prev = fn(data, prev, generator)

        data["pos"] = update["pos"]           # x0 prediction, fed straight back
        if post_step_fn is not None:
            data["pos"] = post_step_fn(data["pos"], raw_t)
        # absent (not stale) on a lean step - see SamplerConfig.lean_steps
        if "aa" in update:
            data["seq"] = update["aa"].argmax(dim=-1)
        if "atom_pos" in update:
            data["atom_pos"] = update["atom_pos"]
        if "aatype" in update:
            data["aatype"] = update["aatype"]
        if "latent" in data and "latent" in update:
            data["latent"] = update["latent"]

        if is_out_step:
            snapshot = {k: data[k] for k in _SNAPSHOT_KEYS if k in data}
            snapshot["step"] = step
            snapshot["raw_t"] = raw_t
            snapshot["result"] = update
            snapshot["aa_logits"] = update["aa"]
            results.append(snapshot)
            # out_steps[-1] is positional, not maximal: an unsorted --out_steps
            # truncates the run (eval_...:672-673). Guarded against an empty
            # list, which inference/sampling.py:90 is not.
            if step == out_steps[-1]:
                break
    return results


def sample_group(step_fn_group: Callable, datas: Sequence[dict],
                 prevs: Sequence[dict], cfg: SamplerConfig, *,
                 generators: Sequence[torch.Generator | None] | None = None,
                 ) -> list[list[dict]]:
    """`sample`, advancing N INDEPENDENT designs in lockstep (N019).

    Returns one snapshot list per design, in the order given - i.e. exactly
    what calling `sample` N times would return, design for design.

    The loop below is `sample`'s, verbatim, with every per-design statement
    moved inside a ``for i in range(n)``. That is the whole trick and it is
    why it is safe: **nothing is shared between designs.** The schedule is a
    function of `step` alone, so `raw_t`, `scaled_t_pos`, `scaled_t_seq`,
    `is_out_step` and the termination test are identical for every member and
    are computed once; everything that carries state - ``pos``, ``prev``,
    ``seq``, the snapshots, and the random draws behind `step_fn_group` - is
    per design. Design *i* sees precisely the inputs it would have seen alone.

    What the caller must supply is a per-design `generator`. The ordinary
    `generate.py` loop draws every design off ONE generator in sequence, and
    that ordering cannot survive being advanced in lockstep (design 0 would no
    longer consume the whole stream before design 1 starts). So the grouped
    entry point derives an independent, seed-derived generator per design -
    which also means ``--group_size 1`` and ``--group_size 8`` are the same
    designs, and that is what makes the identity gate checkable at all.

    The exotic hooks `sample` carries (`post_step_fn` for ``--sym``,
    `final_step_fn` for ``--sample_final``) are deliberately NOT supported
    here: both mutate the step in ways the capture cannot see, and
    `scripts/generate.py` routes those flags to the ordinary path instead.
    """
    n = len(datas)
    if len(prevs) != n:
        raise ValueError(f"{n} data dicts but {len(prevs)} prev dicts")
    gens = list(generators) if generators is not None else [None] * n
    if len(gens) != n:
        raise ValueError(f"{n} data dicts but {len(gens)} generators")

    out_steps = list(cfg.out_steps)
    if out_steps and out_steps != sorted(out_steps):
        warnings.warn(
            f"out_steps {out_steps} is not ascending; upstream breaks on the "
            f"last element as written ({out_steps[-1]}), so later steps are "
            f"never reached. Reproducing that.", stacklevel=2)

    timescale_pos = parse_timescale(cfg.timescale_pos)
    timescale_seq = (None if cfg.timescale_seq is None
                     else parse_timescale(cfg.timescale_seq))

    datas = [dict(d) for d in datas]
    init_prevs = list(prevs)
    prevs = list(prevs)
    denominator = cfg.raw_t_denominator or cfg.num_steps

    results: list[list[dict]] = [[] for _ in range(n)]
    for step in range(cfg.start_steps, cfg.num_steps):
        raw_t = 1 - step / denominator
        scaled_t_pos = timescale_pos(raw_t)
        scaled_t_seq = 1.0 if timescale_seq is None else timescale_seq(raw_t)
        for i in range(n):
            datas[i]["t_pos"] = scaled_t_pos * torch.ones_like(datas[i]["t_pos"])
            datas[i]["t_seq"] = scaled_t_seq * torch.ones_like(datas[i]["t_seq"])
            if raw_t < cfg.prev_threshold:
                prevs[i] = init_prevs[i]

        is_out_step = step in out_steps
        need_aa = is_out_step if cfg.lean_steps else True
        updates, prevs = step_fn_group(datas, prevs, gens, need_aa=need_aa)

        for i in range(n):
            data, update = datas[i], updates[i]
            data["pos"] = update["pos"]
            if "aa" in update:
                data["seq"] = update["aa"].argmax(dim=-1)
            if "atom_pos" in update:
                data["atom_pos"] = update["atom_pos"]
            if "aatype" in update:
                data["aatype"] = update["aatype"]
            if "latent" in data and "latent" in update:
                data["latent"] = update["latent"]

            if is_out_step:
                snapshot = {k: data[k] for k in _SNAPSHOT_KEYS if k in data}
                snapshot["step"] = step
                snapshot["raw_t"] = raw_t
                snapshot["result"] = update
                snapshot["aa_logits"] = update["aa"]
                results[i].append(snapshot)
        if is_out_step and step == out_steps[-1]:
            break
    return results
