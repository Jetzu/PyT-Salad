# N020 · HkLinear matmul lowering: batch-1 bmm -> mm/addmm + inductor matmul templates

type: task · date: 2026-09-26 · status: done

## Goal

`HkLinear.forward` was `torch.einsum("...k,ki->...i", x, self.w)`. ATen lowers
that to `bmm` with batch=1, and N017 verified there was not a single `mm` or
`addmm` in the whole generated module. Consequences measured before this item:
cuBLAS is 62 % of GPU-busy time at ~5.98 us per launch for problems whose
arithmetic is under 1 us; `bmm` forecloses inductor's matmul templates and their
bias-epilogue fusion; N016 measured those templates 1.21-1.37x faster than
cuBLAS on the narrow projections and BIT-IDENTICAL at fp32, while the wide ones
(`dist`, `proj_out`) are slower on Triton - so this wants PER-SHAPE selection.

Second motivation (from the user): the model must run on consumer GPUs and
support Blackwell sm_120. An inductor template is autotuned per device at
compile time; a cuBLAS batch-1 `bmm` is a fixed library heuristic.

## Plan

One variable per arm, `SALAD_OPT_MM` x `SALAD_OPT_GEMMTPL`:

| arm | HkLinear | templates |
|---|---|---|
| base | einsum -> batch-1 bmm | off (= HEAD fd9ba32) |
| mm | fold leading axes, `torch.mm` + bias add | off |
| mmtpl | as `mm` | on |
| addmmtpl | fold, `torch.addmm` | on |
| einsumtpl | einsum | on (CONTROL: inductor has a `tuned_bmm`, so this separates "templates" from "mm lowering") |

Gate: `tests/test_model_parity.py` x 4 configs + geometry/stack/encoder/sampling,
PLUS the bitwise controls, because the Angstrom floor is not sufficient for a
change to how every matmul lowers (N017 caught `TORCHINDUCTOR_FREEZING=1` at
pos 8.8e-06, under the floor and not exact):
  * CPU, compiled `forward_core`, all eleven outputs, L=64 and L=100, max|d| = 0
  * H100 under `SALAD_OPT_DETSEG=1` (bitwise there too) - the CPU control cannot
    see the GPU kernel SELECTION at all, since CPU has no Triton templates.

## Changes

This agent took over after the previous one died on a rate limit, having
produced most of the data (all `cpu_*`/`gpu_*` logs and sbatch scripts except
the templates-specific bitwise control) but no decision and a wrong default.
Fixed and closed:

* `salad_torch/modules/basic.py` - `HkLinear.forward` now branches on
  `opt.MM_MODE` (`einsum`/`mm`/`addmm`), ported unchanged from the snapshot.
* `salad_torch/opt.py` - `SALAD_OPT_MM` default fixed to `einsum` (was
  wrongly `mm` in the handoff - `mm` is not bit-exact, see Evidence). Added
  `MM_MODE`, `GEMM_TEMPLATES` (default `False`, was wrongly `True`). Corrected
  two now-falsified docstring claims: "`mm` is bit-identical to `einsum`"
  (measured 5.6e-05, not 0) and "templates are only reachable via the `mm`
  rewrite" (refuted - inductor's `tuned_bmm` reaches them at `MM_MODE=einsum`
  unchanged).
* `salad_torch/modules/fast.py` - `PRESETS["exact"|"fast"|"deterministic"]`
  all carry `gemm_templates=False,` explicitly (not silently omitted), each
  with a comment pointing at the rejection evidence; the `__init__`
  `_ind.max_autotune_gemm = bool(p.get("gemm_templates")) and
  opt.GEMM_TEMPLATES` wiring is kept live (not deleted) so a future validated
  `fast`-only landing is a one-line change. Applied as 4 surgical hunks
  against the LIVE file (not a wholesale copy), since N021's uncommitted
  `batch_size`/packed-batching work shares this file - see RESULTS.md
  section 10.
* Added `x_cpu_bits.py`'s missing `SALAD_OPT_GEMMTPL` wiring (the prior
  agent's CPU jobs only ever varied `SALAD_OPT_MM`), `cpu_tplcontrol.sbatch`,
  `gpu_tplcontrol.sbatch` - the templates-specific bitwise controls the prior
  agent's jobs never ran (they checked `mm`/`addmm`+templates bitwise but not
  `einsum`+templates, the combination that ships once the default is fixed).

## Evidence

Full tables and job logs in `results/N020-matmul-lowering/RESULTS.md`
(sections 1-9). Headline numbers:

* **`einsum -> mm`/`addmm` lowering: REJECTED.** Not bit-exact (CPU, pos
  max|d| 5.58e-05 @ L=64 / 4.20e-05 @ L=100, both above the 2e-05 A floor);
  buys -0.3 % on its own with the cuBLAS launch count UNCHANGED (301 -> 301);
  `addmm` additionally FAILS `test_model_parity` outright on `multimotif_vp`
  (jobs 904334/904335/904356/904374).
* **Matmul templates alone (`einsum` unchanged): -14.0 % at L=100 (3.071 ->
  2.643 ms), -6.4 % at L=400 (7.079 -> 6.625 ms)** - the win erodes with
  length like P7 did, though less severely (~2.2x shrink vs P7's ~4x). Cause:
  the narrow fixed-shape GEMMs templates win on don't scale with L, so the
  absolute time saved stays ~constant (~470 us) while the rest of the step
  grows (jobs 904339/904341, 3 reps each, <0.1% spread).
* **Deliverable 1, the deciding number: NOT bit-exact.** CPU control (job
  904402): templates are a no-op on CPU, trivially bit-identical (confirmed,
  not assumed). GPU `SALAD_OPT_DETSEG=1` control (job 904395, the one the
  prior agent's jobs never ran for `einsum`+templates specifically): pos
  max|d| = **5.84e-05 @ L=100** / **7.87e-04 @ L=400**, both above the 2e-05 A
  floor. Reinforced two ways: (a) even a same-config cold-cache RERUN under
  DETSEG isn't bit-identical at L=400 (job 904338's `einsum_tpl0_rerun`,
  pos 1.37e-03, 1 discrete `aatype` flip on a pure no-op recompile - a
  methodology confound noted but not chased further); (b) three independent
  cold-cache compiles of the SAME templates-on config disagree with each
  other by up to pos 8.87e-05 (job 904348) - the autotune benchmark is a
  timing race at shapes where candidates are close, so the config is not
  even reproducible against itself.
* **Deliverable 2, parity suite with `SALAD_OPT_GEMMTPL=1`: `SUITE_EXIT=0`**
  on all four configs + geometry 38/38 + stack/encoder/sampling (jobs
  904396, re-confirmed 904646/904649 on the fully-fixed code, and 904659 on
  the live git tree). Caveat stated explicitly in RESULTS.md: this suite is
  eager/CPU-only and structurally cannot see a compile-time/GPU-only change -
  real evidence, but not sufficient on its own, exactly as the task brief
  warned.
* **Portability**: `grep` confirms nothing pins `BLOCK_SIZE`/`num_warps`/
  `num_stages`; `max_autotune_gemm_backends="ATEN,TRITON"` keeps ATEN as a
  fallback, observed degrading gracefully live in the autotune logs
  (`Exception out of resource: shared memory ...` for specific candidates,
  survivors still picked). Not tested on Blackwell/sm_120 - this torch
  (2.5.1+cu121) cannot target it; flagged, not measured.

## Outcome

**Templates do NOT land in `exact` or `deterministic`** (not bit-exact - the
one bar those presets require) **and do NOT land in `fast` either** (no n=160
structural validation has been run, and this project has already produced
"confident wrong answers" at n=8/n=40 twice - N011, N015 - so a 6-14 %
numeric-perturbing win doesn't ship into a default preset without that
check). `SALAD_OPT_GEMMTPL` stays available as an opt-in debug/experimentation
flag, default `False`, requested by no shipped preset.

`SALAD_OPT_MM` defaults to `einsum` (fixed - the handoff had it wrongly
defaulting to `mm`, which is not bit-exact). `mm`/`addmm` remain as rejected,
opt-in-only bisection knobs.

Net effect on the shipped model: **none** in any default configuration - the
default preset behaviour is bit-identical to pre-N020 HEAD (same `einsum`
lowering, no templates). The entire value of this item is the measurement
and the `SALAD_OPT_MM`/`SALAD_OPT_GEMMTPL` infrastructure kept live for a
future, properly-validated `fast`-only landing.

Full parity gate re-run directly on the live git tree after porting
(job 904659): `SUITE_EXIT=0`, on top of live HEAD (which had moved to
`7eb9a10` during this item's run - N022's VRAM-audit commits plus an
independent fix for an unrelated `_cse_graph_pass` pickling bug this item's
own logs had also flagged).

No separate Codex red-team pass was run for this item (see Next) - the
shipped change is a no-op by default, and the bitwise CPU+GPU-DETSEG controls
in Evidence serve the same function this project's convention assigns to
"control" for this exact class of question (N013, N017, N019 all used the
identical method).

## Outputs

`.agent-research/results/N020-matmul-lowering/` - `RESULTS.md` (full write-up,
10 sections), all sbatch scripts and job logs (prior agent's + this agent's:
`cpu_tplcontrol_904402.log`, `gpu_tplcontrol_904395.log`, `gate_tpl_904396.log`,
`gate_final_904646.log`, `gate_default_904649.log`, `gate_live_904659.log`),
`x_*.py` benchmark/comparison scripts. Snapshot
`/hnvme/workspace/v181bb10-salad/n020` (~160 inodes) left in place per the
task's own "deliver from snapshot" framing.

## Next

* An n=160 structural validation of `fast` + `gemm_templates=True` (N015's
  method - generate ~160 designs each arm, compare SSE/Rg/clash
  distributions) would let a future item land templates in `fast`
  specifically. Real win (6-14%), just not proven safe to ship alone.
* This item found that `SALAD_OPT_DETSEG=1` does not guarantee bit-identical
  output across independent COLD-CACHE recompiles at L=400 (a discrete
  `aatype` flip on a literal no-op rerun, `gpu_bits_904338.log`). N013/N019's
  own "bit-identical under DETSEG" claims were established within a single
  compiled process, at L=100/L<=100 - this doesn't contradict them, but it
  is a boundary condition worth a dedicated look before leaning on DETSEG
  reproducibility at length in a future item. Not chased further here
  (out of scope for closing N020).
* No red-team (Codex) pass was run for this item, breaking from the standard
  four-role workflow, given time/resource constraints and that the shipped
  change is inert by default. If a future agent lands `fast`+templates per
  the first bullet, that landing should get the standard red-team pass this
  item's REJECTION did not.
