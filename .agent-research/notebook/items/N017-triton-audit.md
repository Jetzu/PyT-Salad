# N017 · Triton kernel audit: the 524 unaudited inductor kernels

type: research · date: 2026-09-26 · status: done

## Goal
Attribute the 524 Triton kernels of one captured replay of the compiled
denoiser core to source and stage, rank the fusion opportunities, prototype and
MEASURE the top ones. All prior speed work (P1-P7) went at the 321 cuBLAS
launches; the Triton half had never been looked at.

## Plan
1. Profile the lean captured step with `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`
   (without it every inductor kernel reports as the literal name `triton_` and
   the profile is one undifferentiated row) plus `TORCH_COMPILE_DEBUG=1`, then
   join profiler name -> `output_code.py` kernel definition -> its `# Graph
   fragment` node list -> `fx_graph_readable.py` `# File:` annotation.
2. Measure the marginal cost of ONE kernel inside a captured graph, so
   "remove a kernel" has a price.
3. Prototype behind env switches in a PRIVATE snapshot; measure whole-forward
   lean-step ms, one arm per process, one length per process.

## Changes
Nothing landed in the shared tree (by instruction). Delivered as patches:
`results/N017-triton-audit/O1_graph_cse.patch` (recommended),
`O2_ipa_contractions.patch` (measured, fails the sampling gate),
`O4_index_mean_REJECTED.patch` (measured, negative - kept as the record).
Working snapshot `/hnvme/workspace/v181bb10-salad/n017live`, md5s in
`results/N017-triton-audit/live_snapshot_md5.txt`.

## Evidence
| job | partition | what |
|---|---|---|
| 903957 | h100 | first profile + inductor debug dump (HEAD a8533a7) |
| 903972 | h100 | same, unique kernel names + the launch-floor probe |
| 903992 | h100 | arms base / index_mean / fx_graph_cse / freezing / combined |
| 903993 | h100 | attribution on the live snapshot (P7 included) |
| 904001 | cpu 48 | bitwise control, L=64 and L=100 |
| 904010 | cpu 48 | full parity gate x 3 arms (8 tests each) |
| 904079 | h100 | O2 timing + CSE at L=50/100/200 |
| 904080 | cpu 48 | bitwise control for `freezing` vs the narrow CSE |

## Outcome
**Reproduced the stated baseline exactly** (921 kernels = 524 triton + 359
gemm + 36 memcpy, 3432 us) on committed HEAD, and re-measured it on the live
tree where P7 has landed uncommitted (727 = 381 + 308 + 36, 3020 us).

**The economics, measured, not assumed:** a Triton kernel inside a captured
graph costs **1.06 us** just to exist (no-op kernel replayed 0/50/100/200/400
times); a cuBLAS launch averages **5.98 us**. 63 % of the Triton launches (331
of 524) carry exactly one ATen op. So kernel COUNT is the currency, and any
transformation that trades a Triton kernel for a GEMM launch loses ~5x.

**Root finding: inductor has no CSE pass in its inference pipeline**, and the
denoiser rebuilds the same Gram-Schmidt frame **61 times per forward** (46 on
the live tree). 31 of those are exact duplicates.

**O1 SHIPPED-READY: `fx_graph_cse` via `joint_custom_post_pass`.** -2.9 / -2.7 /
-2.5 % of a lean step at L=50/100/200, exactly -50 kernels at every length,
**bit-identical** on CPU at two lengths in two independent jobs (max|d| = 0.0).
Pure FX-graph rewrite: no Triton written, no tuned constants, PORTABLE to
sm_80/86/89 and sm_120.

**`TORCHINDUCTOR_FREEZING=1` is a trap** - same speed, same kernel count, runs
the same CSE, but ALSO folds parameters and layouts: pos 8.8e-06, raw_angles
1.5e-03. Under the 2e-05 A parity floor, so it would PASS `test_model_parity`
while not being fp32-exact. The documented knob is the wrong answer here.

**O2 (IPA einsum -> mul+sum) measured -3.1 %, composes with O1 to -5.9 %** by
removing 48 cuBLAS launches for 24 Triton ones - the launch economics running
the right way. But it **fails `test_sampling_parity`** (7/64 residues, 40.8x
floor), same signature as DETSEG, so it is a deviating-preset candidate only
and would need N015-style n=160 validation first.

**O4 (index_mean's (N,F) ones_like -> (N,1)) is bit-exact and measurably
WORSE**: +9 kernels, +0.003 ms. Inductor was already folding the wasted
columns; the narrower buffer stops fusing with its (N,F) neighbours.

**What is left:** 59 % of the remaining GPU-busy time is cuBLAS launch latency.
Every `HkLinear` is `einsum("...k,ki->...i")`, which ATen lowers to **bmm with
batch=1** - there is not one `mm` or `addmm` in the generated module, which
forecloses inductor's matmul templates, its per-device autotuning and its bias
epilogue. That, not the Triton half, is where the next 2x is.

## Outputs
`.agent-research/results/N017-triton-audit/` — `README.md` (the full report),
`attribution_table.txt` + `live/attribution_table.txt`, `kernel_rows.json`,
`profile_lean.json`, the three patches, `inductor/` + `live/inductor/`
(inductor's generated code and FX graphs, gzipped), and the harness
(`x_attrib.py`, `x_parse.py`, `x_bench.py`, `x_cpu_bits.py`, the sbatch files).

## Next
- T: land O1 (patch is self-contained, guarded, falls back on ImportError).
- T: the durable form of O1 is source-level frame threading in
  `blocks.py`/`features.py`/`attention.py` — needs no torch-private API. Its
  ceiling is the -2.7 % above; it is NOT separately measured.
- T: `HkLinear.forward`'s einsum -> `matmul`, to get `mm`/`addmm` instead of 301
  batch-1 `bmm`s. Biggest remaining item and it is a portability win too
  (inductor templates autotune per device; cuBLAS batch-1 bmm does not).
- T: re-measure P7 on a consumer card before trusting its -9.2 % — it traded
  launches for bandwidth on a 6x-tiled axis, which is the right trade at
  3.35 TB/s and a worse one at ~1 TB/s.
