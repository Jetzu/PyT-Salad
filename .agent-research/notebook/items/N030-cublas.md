# N030 · cuBLAS: bit-exact deterministic matmul templates + GEMM launch reduction

type: task · date: 2026-09-27 · status: done

(The brief called this "N027"; N027/N028/N029 were already allocated to the two
concurrent sibling agents by the time this item was created, so it is N030.)

## Goal

Attack the cuBLAS side - 301 launches, 1877 us, 62 % of GPU-busy time in the
captured lean step at L=100 - which averages 5.98 us per launch for arithmetic
under 1 us. Specifically:

1. Make inductor's matmul TEMPLATES (N020's -14 % at L=100) **bit-exact and
   deterministic**, instead of the timing race N020 correctly rejected.
2. Look for GEMM merges that do not go through P7's miscompiled code path.
3. Census the 301 launches: are they all necessary GEMMs?

## Plan

Core idea (new): replace "fastest candidate" with **"fastest candidate that is
BITWISE identical to cuBLAS at this shape, measured on the machine that is
compiling"**. That is self-verifying, so it needs no per-shape allowlist, no
tuned constant and no arch guard, and it removes N020's two objections at once:
the output cannot depend on which candidate wins the timing race, because every
survivor produces the same bits.

## Changes

* NEW `salad_torch/gemm_exact.py` - monkeypatches
  `torch._inductor.kernel.{mm,bmm}.autotune_select_algorithm` to filter matmul
  TEMPLATE candidates down to those bitwise-equal to cuBLAS at that shape
  (`mode="exact"`) or to a deterministic tie-break among the fastest
  (`mode="stable"`). Research/debug knob only - see Outcome.
* `salad_torch/opt.py` - `GEMM_TPL_MODE`/`resolve_gemm_templates` (mode
  resolution for `SALAD_OPT_GEMMTPL=off|all|exact|stable`) and
  `DECOMPOSE_SMALL_BMM`/`resolve_decompose_small_bmm` (the
  `SALAD_OPT_DECOMPBMM` threshold/veto for inductor's `decompose_mm_pass`).
* `salad_torch/modules/fast.py` - `PRESETS["fast"]["decompose_small_bmm"] =
  32` (the only preset-default CHANGE); `exact`/`deterministic` explicitly
  `decompose_small_bmm=False`; `FastDenoiser.__init__` wires
  `gemm_exact.install/uninstall` (salted into `joint_custom_post_pass`'s NAME
  so the FX-graph cache can't serve one mode's artifact to another) and
  `decompose_mm_pass`'s `post_grad_fusion_options`.
* Landed as a hand-built patch against committed `0915f5a`, applied via
  `git stash`/`git apply`/`git stash pop` around a concurrent sibling's
  (N028) uncommitted WIP in the same two files - see RESULTS.md section 9.

## Evidence

Full detail in `.agent-research/results/N030-cublas/RESULTS.md`. Headline
numbers:

* Per-shape bitwise-verified template filter (`gemm_exact.py mode="exact"`):
  **NOT bit-exact end to end** (pos max\|d\|=5.44e-05 A vs a 2e-05 floor,
  barely better than the unfiltered arm's 5.84e-05) and **not even
  reproducible against itself** across 3 independent cold-cache compiles
  (rep1 vs rep3 pos 3.46e-05, rep1 vs rep2 exactly 0.0) - the candidate SET
  surviving verification drifts between compiles, not just which survivor
  wins the speed race. -11.0% at L=100 for a 657-2500s compile tax.
* `decompose_small_bmm=32` (inductor's `decompose_mm_pass`, NOT an autotune
  pass): **-15.5%** at L=100 on the current P7-re-enabled baseline (3.0721 ->
  2.5946 ms lean, job 907765), holds up almost exactly against the pre-P7
  measurement of -16.6% (P7 and this do not meaningfully compete for the
  same GEMMs). NOT bit-exact (pos max\|d\|=4.96e-05 A, job 907766) so never
  `exact`/`deterministic`, but **perfectly reproducible across 3 cold
  compiles** (job 907764, unlike the templates) and clears the n=160
  design-quality gate on the ACTUAL shipped `fast` preset (job 907763: 0/160
  diverged; final re-run job 907993 post-rebase: 1/160 = 0.6%, comfortably
  under the 10% limit).
* Final pre-commit gate on the exact code committed (N028's concurrent WIP
  stashed out): CPU (`salad_jobs/n030_final_cpu.sbatch`, job 907992)
  `CPU_SUITE_EXIT=0`; GPU (`salad_jobs/n030_final_gpu.sbatch`, job 907993)
  `GPU_SUITE_EXIT=0` (parity suite + quality fast n=160 + quality exact n=64
  + `--group_size auto` sanity, all `EXIT 0`).

## Outcome

**Neither of the brief's two proposed routes ships.** Pinning the autotune
and the per-shape bitwise allowlist are both refuted with numbers (not just
argued): the allowlist idea is self-defeating because the candidate SET
itself, not just the tie-break, is unstable across compiles. **A third route
this item found instead - `decompose_small_bmm` (inductor's
`decompose_mm_pass`, a config-based FX rewrite with no autotune at all) -
ships in `fast`**: -15.5% (the single biggest cuBLAS-side win found, second
only to P7 itself among ALL launch-reduction work on this model), no compile
tax, reproducible, gated by both suites on the actual code state.

## Outputs

* `.agent-research/results/N030-cublas/RESULTS.md` - full writeup, all
  sections.
* `.agent-research/results/N030-cublas/` - job scripts (`j1`-`j10`), harness
  (`x_*.py`), and every raw log this item's numbers come from.
* `salad_jobs/n030_final_cpu.sbatch` / `n030_final_gpu.sbatch` - the
  pre-commit gate, reusable for the next item that touches this code.
* Commit hash: see INDEX.md row.

## Next

* Re-measure `decompose_small_bmm`'s marginal contribution once N028's
  `frame_mulsum` lands - both target the same ~90 3x3-frame `bmm`s by
  different mechanisms and the win will likely shrink, not stack (RESULTS.md
  section 9).
* The mechanism behind the per-shape filter's own cold-compile instability
  (section 6) was measured, not root-caused - out of scope once the ship
  decision was NO regardless.
* `gemm_exact.py mode="stable"` exists and works (deterministic tie-break,
  never claimed cuBLAS-exactness) but was not separately validated at n=160;
  moot for `exact`/`deterministic` and slower than `decompose_small_bmm` for
  `fast`, so not pursued further.
