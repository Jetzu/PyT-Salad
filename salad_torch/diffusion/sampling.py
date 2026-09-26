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

__all__ = ["SamplerConfig", "sample", "sample_group", "make_initial_data",
           "pack_designs", "split_packed_snapshot"]

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



# ---------------------------------------------------------------------------
# N021 - PACKED batching: B designs concatenated along the flat residue axis
# ---------------------------------------------------------------------------
#: Conditioning fields that are dense ``(N, N)`` pair maps. Packing these is
#: NOT a concatenation - they have to become BLOCK DIAGONAL, so design `i`'s
#: map covers only design `i`'s residues and every cross-design entry is the
#: unconditional sentinel (0 / False). Concatenating them along axis 0 would
#: produce a ``(B*L, L)`` tensor, i.e. the wrong shape, and broadcasting one
#: design's map across the pack would apply its motif to every member.
_PAIR_FIELDS = ("dmap", "dmap_mask", "omap", "chain_contacts")
#: Fields the packer knows how to handle by a rule other than "concatenate".
_SPECIAL_FIELDS = ("batch_index", "dssp_mean", "dssp_mean_mask", "cloud_std")


def _block_diagonal(values: Sequence[Tensor], lengths: Sequence[int]) -> Tensor:
    """Place each ``(L_i, L_i, ...)`` map on the diagonal of one ``(N, N, ...)``."""
    total = int(sum(lengths))
    v0 = values[0]
    out = torch.zeros((total, total) + tuple(v0.shape[2:]),
                      dtype=v0.dtype, device=v0.device)
    off = 0
    for v, n in zip(values, lengths):
        out[off:off + n, off:off + n] = v
        off += n
    return out


def pack_designs(datas: Sequence[dict], prevs: Sequence[dict]
                 ) -> tuple[dict, dict, list[int]]:
    """Concatenate B per-design ``(data, prev)`` pairs into ONE packed pair.

    salad has **no leading batch dimension**. A batch is several proteins
    concatenated along the flat residue axis with a companion
    ``batch_index``/``chain_index``/``mask``; every segment op (`index_mean`,
    `index_sum`, `extract_neighbours`) expects exactly that, so
    `torch.stack` is the wrong collate and `salad_torch.data.collate`'s
    `concat_collate` (which reproduces upstream's `np_collate`) is the right
    one. This is the inference-side counterpart of it.

    **Nothing is padded.** `notes/04_PARITY_AND_PERF.md` B1: a `-1`
    "no neighbour" sentinel writes the *original* finite distance back into
    column N-1, so a protein sitting next to masked padding, or at any offset
    other than the end of the flat axis, gets a measurably different answer.
    That bug is upstream's and is reproduced bit-for-bit by this port, so the
    only defence is to concatenate exact-length designs and never pad. A
    packed design is therefore **not** bit-identical to the same design
    generated alone - see `.agent-research/results/N021-packed-batching/`
    for how far apart they actually come out.

    Returns ``(packed_data, packed_prev, lengths)``; `lengths` is what
    `split_packed_snapshot` needs to take the pack apart again.

    Every field is packed by an explicit rule and an unrecognised key RAISES.
    A silent fallback is exactly how a conditioning channel would end up
    broadcast across the whole pack.
    """
    n = len(datas)
    if len(prevs) != n:
        raise ValueError(f"{n} data dicts but {len(prevs)} prev dicts")
    lengths = [int(d["mask"].shape[0]) for d in datas]
    if n == 1:
        # B=1 must be the ORDINARY computation, byte for byte: same dict, same
        # tensors, no renumbering (batch_index is already all-zero).
        return dict(datas[0]), dict(prevs[0]), lengths

    keys = set(datas[0])
    for i, d in enumerate(datas[1:], 1):
        if set(d) != keys:
            raise ValueError(
                f"design {i} carries a different set of keys than design 0 "
                f"({sorted(set(d) ^ keys)}); a packed batch must be "
                f"homogeneous - run heterogeneous conditioning unpacked")

    out: dict[str, Any] = {}
    for k in keys:
        values = [d[k] for d in datas]
        v0 = values[0]
        if k == "batch_index":
            out[k] = torch.cat([torch.full_like(v, i)
                                for i, v in enumerate(values)])
        elif k == "chain_index":
            # RENUMBERED, not concatenated - and this is not cosmetic.
            # `batch_index` separates the members in every op that takes
            # `same_item`, but two of salad's reductions key on `chain_index`
            # ALONE and would fuse members that both call their first chain 0:
            #   * `modules/blocks.py` `Update`'s per-chain global gate,
            #     `index_mean(chain_gate * local_update, chain, mask)`, in
            #     EVERY block of the trunk;
            #   * `geometry/features.py`'s relative-position feature, whose
            #     chain length is `index_count(chain, ones)`, so a shared id
            #     reports a B*L-residue chain.
            # Measured on CPU (deterministic) at L=100, B=2, one forward with
            # identical injected randomness: leaving the ids colliding moves
            # `local` by 34-38 and `pos` by ~4 A, i.e. it is a different
            # protein, and it moves the LAST member too - so it is not the B1
            # sentinel. Only chain-relative comparisons are ever made
            # (`chain[:, None] == chain[None, :]`), so a per-member offset is
            # invisible within a member.
            #
            # NB upstream's own training collate (`dataset.py` step 8,
            # `_numerical_chain_index` per part) does NOT do this, so packed
            # TRAINING batches in salad share chain ids across items and take
            # both of these leaks. Out of scope here, but recorded.
            offset, renumbered = 0, []
            for v in values:
                renumbered.append(v + offset)
                offset += int(v.max().item()) + 1
            out[k] = torch.cat(renumbered)
        elif k in _PAIR_FIELDS:
            out[k] = _block_diagonal(values, lengths)
        elif k == "dssp_mean":
            # a 1-D (3,) request means "this composition, for every residue",
            # and `prepare_condition` expands it over the WHOLE flat axis - so
            # packing it as-is would apply design 0's composition to all B.
            # Expand each design's request to its own (L_i, 3) rows first.
            rows, masks = [], []
            for v, L_i in zip(values, lengths):
                if v.dim() == 1:
                    rows.append(v[None].expand(L_i, v.shape[0]).contiguous())
                    masks.append(torch.ones((L_i,), dtype=torch.bool,
                                            device=v.device))
                else:
                    rows.append(v)
                    masks.append(None)
            out[k] = torch.cat(rows, dim=0)
            if any(m is not None for m in masks):
                built = [m if m is not None else
                         datas[i]["dssp_mean_mask"]
                         for i, m in enumerate(masks)]
                out["dssp_mean_mask"] = torch.cat(built, dim=0)
        elif k == "dssp_mean_mask":
            continue          # written by the "dssp_mean" branch above
        elif k == "cloud_std":
            if torch.is_tensor(v0) and v0.dim() >= 1 and v0.shape[0] == lengths[0]:
                out[k] = torch.cat(values, dim=0)
            else:
                # a python/0-d scalar: one cloud std for the whole pack. Only
                # safe if every design asked for the same one.
                scalars = [float(v) for v in values]
                if len(set(scalars)) != 1:
                    raise ValueError(
                        "packed designs ask for different scalar cloud_std "
                        f"{sorted(set(scalars))}; pass it per residue or run "
                        "them unpacked")
                out[k] = v0
        elif torch.is_tensor(v0):
            if v0.shape[0] != lengths[0]:
                raise ValueError(
                    f"do not know how to pack {k!r}: leading dimension "
                    f"{v0.shape[0]} is not this design's residue count "
                    f"{lengths[0]}")
            out[k] = torch.cat(values, dim=0)
        else:
            raise ValueError(f"do not know how to pack non-tensor field {k!r}")

    packed_prev = {k: torch.cat([p[k] for p in prevs], dim=0) for k in prevs[0]}
    return out, packed_prev, lengths


def split_packed_snapshot(snapshot: dict, lengths: Sequence[int]) -> list[dict]:
    """Inverse of `pack_designs` on one snapshot: B per-design snapshots.

    Every tensor whose leading dimension is the pack's total residue count is
    sliced; ``batch_index`` is reset to all-zero so a split snapshot is
    indistinguishable from an unpacked one downstream. Pair maps are diced on
    both axes. Scalars (``step``, ``raw_t``) are copied.
    """
    total = int(sum(lengths))
    offsets, off = [], 0
    for n in lengths:
        offsets.append((off, off + n))
        off += n

    def cut(key, value, lo, hi):
        if isinstance(value, dict):
            return {k: cut(k, v, lo, hi) for k, v in value.items()}
        if not torch.is_tensor(value) or tuple(value.shape[:1]) != (total,):
            return value
        out = value[lo:hi]
        # the second axis is diced ONLY for the known pair maps. Sniffing
        # `shape[1] == total` instead would mis-slice `pos` whenever the flat
        # axis happens to equal the atom count.
        if key in _PAIR_FIELDS and value.dim() >= 2:
            out = out[:, lo:hi]
        return out

    parts = []
    for lo, hi in offsets:
        part = {k: cut(k, v, lo, hi) for k, v in snapshot.items()}
        if torch.is_tensor(part.get("batch_index")):
            part["batch_index"] = torch.zeros_like(part["batch_index"])
        parts.append(part)
    return parts

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
