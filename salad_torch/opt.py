"""Inference fast-path switches (N009, executing the N007 PLAN).

Every switch here defaults to **on** and is either bit-exact or
reassociation-only; the environment variables exist so one benchmark job can
measure each proposal separately in its own process. They are read **once, at
import time**, so they are ordinary Python constants by the time Dynamo traces
anything - a mutable flag would become a guard and force a recompile.

| flag | proposal | class |
|---|---|---|
| ``SALAD_OPT_GATHER=0`` | P3 one-hot x matrix -> row gather | bit-exact |
| ``SALAD_OPT_HOIST=0``  | P6 hoist loop-invariant neighbour stages | bit-exact |
| ``SALAD_OPT_NOPC=0``   | P4+P5 skip the dead pair-conditioning work | P4 bit-exact, P5 reassociation |
| ``SALAD_OPT_LEAN=0``   | P2 skip the AA decoder + angle head off-step | skip (see below) |

P4/P5 additionally carry a **runtime** gate: they only fire when the design has
no pair conditioning at all (``dmap_mask`` all false AND ``flags`` all false),
which `FastDenoiser` decides per design from the real conditioning tensors.
The switch below can only ever turn that off, never on.
"""
from __future__ import annotations

import os

__all__ = ["GATHER_ONEHOT", "HOIST_NEIGHBOURS", "SKIP_PAIR_COND", "LEAN_STEPS",
           "NEED_AUX", "BATCH_COND", "DET_SEGMENT", "AGGRESSIVE_FUSION",
           "MM_MODE", "GEMM_TEMPLATES"]


def _flag(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


#: P3 - `sequence_relative_position(one_hot=True) @ W` and `one_hot(aa) @ W`
#: become `W[idx]`. Bit-exact: 65 of the 66 products are exactly 0.0 and
#: `a + 0.0 == a` in IEEE-754 for any finite a, in any summation order.
GATHER_ONEHOT = _flag("SALAD_OPT_GATHER")

#: P6 - compute the sequence-index neighbour stage once instead of 12 times,
#: the self-conditioning branch's distance matrix + index + spatial stages once
#: instead of 6 times, and share the CA distance matrix between the spatial and
#: random stages. Pure common-subexpression elimination, bit-exact.
HOIST_NEIGHBOURS = _flag("SALAD_OPT_HOIST")

#: P4 + P5 - with no pair conditioning the `dmap`/`flags` pair terms are exactly
#: zero and the fourth ("block contact") neighbour stage returns all -1, so the
#: condition branch runs at K=48 instead of K=64. P4 is bit-exact; P5 changes
#: only the reduction tree over K (the dropped slots contribute exactly 0.0).
SKIP_PAIR_COND = _flag("SALAD_OPT_NOPC")

#: P2 - let the sampler ask for the sequence heads only on reporting steps.
#: A skip, not a reassociation: `pos`/`local`/`trajectory` come out of the same
#: trunk. The one thing it does NOT control is whether inductor compiles the
#: two variants' shared trunk into the same kernels, which is why the gate for
#: shipping it is a byte-identical PDB from a full run, not a tensor diff.
LEAN_STEPS = _flag("SALAD_OPT_LEAN")

#: The dssp head and the distogram neighbour draw, neither of which sampling
#: reads. Default ON - removing them was measured SLOWER once (6.48 -> 6.9 ms),
#: back when the forward was 30 broken Dynamo graphs. `SALAD_OPT_AUX=0`
#: re-tests it against the single-graph forward.
NEED_AUX = _flag("SALAD_OPT_AUX")

#: P7 (N013) - compute the condition branch's `pair_2` for all six blocks in
#: one batched prologue (`modules/cond_prologue.StackedCondPair`). Worth
#: -9.2 % of a lean step at L=100. The feature tensors are bit-identical (the
#: builders are elementwise on a tiled residue axis); the seven projections and
#: the two MLP layers become `bmm(batch=6)` instead of six `mm`s, which is a
#: reassociation - the same class as P5, and a weaker one.
#: Never taken for `multi_motif` (its `pair_2` reads `pos`, not `prev_pos`)
#: nor in `train()` mode (the stacked weights are a cache of the parameters).
#: `SALAD_OPT_BATCHCOND=0` turns it off; a preset can only turn it off too.
BATCH_COND = _flag("SALAD_OPT_BATCHCOND")

#: N013 item 4 - a DETERMINISTIC segment reduction, so `--seed` pins a CUDA
#: run. `index_sum`'s `scatter_add` is an atomicAdd whose ordering varies at
#: ~1.6e-7, and the neighbour argsort amplifies that into a discrete tie flip
#: (notes/04 section 8). Replacing it with a one-hot matmul is deterministic by
#: construction and needs no process-global mode. It CHANGES THE SUMMATION
#: ORDER, so it is its own parity baseline - never the default.
DET_SEGMENT = _flag("SALAD_OPT_DETSEG", default=False)

#: N013 - inductor's `aggressive_fusion` scheduler knob. It merges more nodes
#: into one kernel and is worth -1.4 to -2.4 % of the step. Measured
#: BIT-IDENTICAL twice over: on CPU (deterministic by construction, L=64 and
#: L=100, all eleven outputs, `x_cpu_bits.py`) and ON THE H100 under the
#: deterministic segment reduction, which removes the only source of CUDA
#: run-to-run noise (`cmp_floor.py dg_synth_base dg_synth_aggr`: one shared bit
#: pattern). So it needs no noise-floor argument - it changes no element's
#: arithmetic. `SALAD_OPT_AGGRFUSE=0` turns it off.
AGGRESSIVE_FUSION = _flag("SALAD_OPT_AGGRFUSE")


#: N020 - how `HkLinear.forward` contracts. `"einsum"` (the DEFAULT, and the
#: pre-N020 form) is `einsum("...k,ki->...i")`, which ATen lowers to `bmm` with
#: batch=1. `"mm"` instead folds the leading axes into M and calls `torch.mm`,
#: so the op that reaches inductor is `aten.mm`; the bias stays a separate
#: `add`. `"addmm"` folds and calls `torch.addmm` directly.
#:
#:   ``SALAD_OPT_MM=einsum``  the pre-N020 form -> batch-1 bmm  (DEFAULT)
#:   ``SALAD_OPT_MM=mm``      fold + `torch.mm` + separate bias add
#:   ``SALAD_OPT_MM=addmm``   fold + `torch.addmm`
#:
#: REJECTED, both of them - kept only as a bisection/debug knob, never shipped:
#: `mm` is NOT bit-identical to `einsum` (measured: CPU, all eleven
#: `forward_core` outputs, L=64 max|d|(pos)=5.579e-05, L=100
#: max|d|(pos)=4.196e-05 - ABOVE the 2e-05 A parity floor) and it buys almost
#: nothing on its own (3.071 -> 3.063 ms at L=100, -0.3%, cuBLAS launch count
#: UNCHANGED at 301). `addmm` is likewise not bit-identical AND fails
#: `test_model_parity` outright on `multimotif_vp` (see RESULTS.md). `0`/`off`
#: is a synonym for `einsum`, so the flag also reads as the usual veto -
#: setting it to anything is opting OUT of the default, not turning something on.
_MM_ENV = (os.environ.get("SALAD_OPT_MM") or "einsum").strip().lower()
MM_MODE = {"0": "einsum", "off": "einsum", "false": "einsum", "no": "einsum",
           "1": "mm", "on": "mm", "true": "mm", "yes": "mm",
           "einsum": "einsum", "mm": "mm", "addmm": "addmm"}.get(_MM_ENV)
if MM_MODE is None:
    raise ValueError(f"SALAD_OPT_MM must be einsum|mm|addmm, got {_MM_ENV!r}")

#: N020 - inductor's matmul TEMPLATES (`max_autotune_gemm`), i.e. let inductor
#: benchmark a Triton template against cuBLAS at every GEMM shape in the graph
#: at compile time and keep the faster one. Inductor has a `tuned_bmm` template
#: path too, so this is reachable at the DEFAULT `MM_MODE=einsum` (batch-1
#: `bmm`) without any change to how `HkLinear` contracts - measured: `einsum`
#: unchanged + templates on is -14% at L=100 (2.643 vs 3.071 ms, eroding to
#: -6.4% at L=400), a bigger win than the `mm` lowering ever bought and with NO
#: mm/addmm parity cost. (The `mm` rewrite is not a prerequisite for templates;
#: an earlier brief assumed batch-1 `bmm` forecloses them - refuted.)
#: Per-SHAPE and per-DEVICE by construction: the choice is a measurement on
#: the machine that is compiling, not a constant, which is what a consumer-GPU
#: / sm_120 target needs. Costs compile time (one benchmarking pass per
#: distinct shape, ~12 min for this model's shape set).
#:
#: REJECTED from every preset (default False, unlike `AGGRESSIVE_FUSION`)
#: because it is NOT bit-exact - the one thing this flag needs to be to ship
#: in `exact` the way `aggressive_fusion`/`batch_cond_pair` did. Measured
#: whole-`forward_core`, H100, `SALAD_OPT_DETSEG=1` (the bitwise control; CPU
#: cannot see it at all - Triton templates never engage there, confirmed:
#: max|d| = 0.0 on CPU at L=64/100): `einsum` unchanged, templates on vs off,
#: pos max|d| = 5.84e-05 at L=100 (raw_angles 7.97e-03) and 7.87e-04 at L=400
#: (raw_angles 0.104) - both ABOVE the 2e-05 A parity floor. It also is not
#: even reproducible against ITSELF: three independent cold-cache compiles of
#: the identical templates-on config disagreed by up to pos 8.87e-05 at L=100
#: (`gpu_stability.sbatch`), i.e. the autotune benchmark can pick a different
#: winning kernel at a shape where the candidates are within noise, so the
#: bits depend on a timing race even on the SAME device. See RESULTS.md
#: section on N020's deliverable 1 for the full table and the CPU/GPU-DETSEG
#: methodology. `SALAD_OPT_GEMMTPL=1` is left as an opt-in knob for
#: experimentation (e.g. a `fast`-preset candidate pending an n=160
#: structural validation - see N015's method - which has NOT been run) but no
#: shipped preset requests it.
GEMM_TEMPLATES = _flag("SALAD_OPT_GEMMTPL", default=False)
