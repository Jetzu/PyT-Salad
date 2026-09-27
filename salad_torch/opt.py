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
#:
#: ############ DISABLED - IT PRODUCES BROKEN PROTEINS ############
#:
#: P7 is DEFAULT OFF and every preset sets `batch_cond_pair=False`. Turning it
#: on destroys designs: 17.2 % of them at L=100 (n=64) and 20.6 % at n=160, with
#: CA-CA bond lengths of 620-688 A and Rg 520-590. P7 off is 0/160 at CA-CA
#: 3.7632 / Rg 12.94, which is the pre-P7 reference (3.7605 / 13.07) exactly.
#: Localised three ways: a commit bisect (c5ab7a0 0/64 -> 603051e 11/64), the
#: `SALAD_OPT_BATCHCOND=0` veto at n=160, and a 6-arm toggle matrix that clears
#: CSE, `aggressive_fusion`, TF32, CUDA-graph capture and the N019 refactor.
#:
#: P7's OWN ARITHMETIC IS CORRECT. The prologue agrees with the six sequential
#: `pair_2` calls to 8.6e-6 absolute on a real step-30 state, with bit-identical
#: pair masks in all six blocks, on GPU and CPU, and `StackedCondPair` compiled
#: in ISOLATION is equally correct. The fault is an INDUCTOR FUSION MISCOMPILE
#: of the prologue inside `forward_core`: same inputs, plain
#: `torch.compile(mode="default")`, no capture, `pos` is wrong by **9.586 A**
#: against eager - five orders of magnitude above the 2.0e-05 A compile floor
#: measured on the same forward with P7 off. It enters at block 0's `attn_2`
#: (0.368 A) and compounds to 9.59 A by block 5. `max_fusion_size=1` fixes it
#: and every other knob (aggressive_fusion, pattern_matcher, epilogue_fusion,
#: reduce-overhead) leaves it BIT-IDENTICAL, so ONE two-node fusion is enough.
#: torch 2.5.1+cu121, sm_90. Probably an upstream bug; see TODO T9.
#:
#: REFUTED, so nobody retries it: the prime suspect was the prologue's inline
#: LayerNorm (`pair.mean(-1)` + `pair.var(-1, unbiased=False)` + `rsqrt` over the
#: last dim of a `(6, N*K, 64)` tensor) - three separately-fusable reductions, and
#: `F.layer_norm` lowers as ONE op that inductor schedules as a unit, which should
#: have denied the fusion a partner. Rewritten that way and measured with P7 ON:
#: **32/160 still diverged (20.0 %)**. Same maths, parity fine (0.00001 A), no
#: effect on the miscompile. The bad fusion is elsewhere in the prologue graph.
#: (Watch out when re-testing: a preset's `batch_cond_pair=False` ANDs with the
#: env var, so `SALAD_OPT_BATCHCOND=1` alone does NOT turn P7 on - an arm that
#: looks clean may simply not have run P7 at all. Timings identical to three
#: decimals between "on" and "off" is the tell.)
#:
#: WHY EVERY GATE MISSED IT: all four parity tests drive the EAGER path, where
#: P7 is correct - and the eager branch feeds the prologue six IDENTICAL
#: condition-neighbour index sets, while `forward_core` feeds six DISTINCT ones.
#: The compiled path's actual shape had never been tested. `tests/test_design_quality.py`
#: is the gate that catches this class; run it before re-enabling anything here.
#:
#: Do NOT re-enable without: (a) root-causing the fusion, and (b) 0/160 diverged
#: on `tests/test_design_quality.py --n 160`. It was worth -9.3 to -11.0 % of a
#: whole forward at L=100, eroding to -2.3 % at L=400 (N014) - not enough to
#: gamble a silent 20 % design-loss rate on.
BATCH_COND = _flag("SALAD_OPT_BATCHCOND", default=False)

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
