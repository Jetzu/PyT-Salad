# N033 - torch upgrade evaluation: 2.5.1+cu121 -> 2.7.1+cu128

Evaluation only. No code changes to the port. `triad` conda env (torch
2.5.1+cu121, used by every other job in this repo) untouched throughout.

ID note: brief said "N031" / `.agent-research/results/N031-torch-upgrade/`;
N031 was already the sibling "P7 fusion miscompile" item
(`N031-p7-fusion-rootcause`) by the time this item was created, and N032 was
independently pre-claimed by the brief for a concurrent sibling agent. Landed
as **N033** (`new-item.sh` itself returned N032; bumped by hand to avoid a
third collision - same failure mode INDEX already documents for N027/N030).

## 1. Environment: build plan and inode cost (stated before building)

Chosen route: **apptainer `.sif`, built with `--fakeroot` in node-local
`/tmp` on a login node (has internet; H100 compute nodes do not), copied as
ONE file to `/hnvme`** - identical pattern to the BindCraft and ORCA
containers already on this project.

Sizing check done BEFORE building: AST-scanned every import in
`salad_torch/`, `tests/`, `scripts/` and confirmed the port's own gate tests
(`test_model_parity.py`, `test_{geometry,stack,encoder,sampling}_parity.py`,
`test_design_quality.py`) import only `numpy` and `torch` at runtime.
`jax`/`haiku`/`Bio`/`biotite`/`einops`/`scipy`/`dm-tree`/`msgpack-numpy` in the
`triad` env's `pip freeze` are leftovers for the JAX reference code under
`upstream/` (fixture generation, not run by the gate) and the optional
`--template` path (`Bio.PDB`, lazily imported, guarded by a runtime check) -
none of them are needed here. So the new env could be minimal: base
`python:3.11-slim-bookworm` + `torch==2.7.1+cu128` + `numpy==1.26.4` +
`build-essential` (see 1b).

Stated cost before building: 0 inodes on `/home/hpc` (only ~36k headroom
against the 500k cap), 0 *permanent* inodes on node-local `/tmp` (an
apptainer fakeroot build unpacks a full filesystem tree there transiently -
comparable in file count to a conda env - but that disk is not the
inode-capped Lustre/hnvme mount and is wiped when the build finishes),
**+1 inode on `/hnvme`** for the finished `.sif`.

Actual outcome: built in 65-73s (two builds, see 1b), `torch27_cu128.sif`
3.8-4.1 GB, copied to `/hnvme/workspace/v181bb10-salad/torch27_cu128.sif`.
`/hnvme` inode delta: went from 9,601,541 to 9,601,542 of 37,131,382,848
total (0.03% used) - the brief's inode-pressure warning is a **project-level
soft file-count quota** (confirmed live in every job's `JOB_STATISTICS`
footer: `/hnvme ... Filec 73K FileQ 61K FiHaQ 81K` - already 20% OVER its own
soft quota, pre-existing, not caused by this work), not the raw mount, and a
single extra file changes neither. `triad` conda env: read-only throughout,
confirmed via `pip freeze`/`python --version` checks only, never modified.

### 1b. Two build defects found and fixed (both mine, not torch's)

First build (`torch27_cu128.def` v1: `ca-certificates` only) passed every
CPU-side sanity check (plain `import salad_torch.modules.{model,fast,...}`,
`tests/test_geometry_parity.py` 38/38 on CPU) but **every single GPU
`torch.compile` call in the first batch of 4 jobs failed** with:

```
torch._inductor.exc.InductorError: RuntimeError: Failed to find C compiler.
Please specify via CC environment variable.
```

Root cause: Triton's runtime JIT-builds a tiny CPython extension
(`driver.c` -> `cuda_utils.so`, the CUDA driver bridge) on the FIRST kernel
launch of a process, even though Inductor itself only emits Triton/Python
source text. `python:3.11-slim-bookworm` ships Python.h headers but **no C
compiler at all**. This is an environment-construction bug, not a
torch-2.7 finding - fixed by adding `build-essential` to the `%post` section
and rebuilding (73s, `gcc` present in the second image, confirmed via
`apptainer exec ... which gcc`). All 4 jobs were killed after ~2-9 minutes
by this (SLURM job exit 0/0 in both batches since none of my sbatch scripts
`exit $rc` - state must be read from log text, not the SLURM exit code) and
resubmitted clean.

Second defect (mine, in `n033_det.sbatch`): the cross-rep comparison step
(`x_cmp.py`, reused unmodified from N030) imports `x_common.NAMES`, which
reads `os.environ["SALAD_TREE"]` at import time - I set `SALAD_TREE` on the
`x_dump.py` invocations but forgot it on the `x_cmp.py` invocations, so every
comparison step crashed with `KeyError: 'SALAD_TREE'` even though every dump
step had already succeeded. Fixed by prefixing `SALAD_TREE=$SNAP` on those
two lines too.

Both fixes cost one more `apptainer build` (73s) and one more `sbatch`
resubmission; no torch-version finding changed as a result.

## 2. Reproducer verdict: FIXED in 2.7.1

`.../N031-p7-fusion-rootcause/codex/minimal_conditional.py`, run unmodified,
job 908347:

```
torch 2.7.1+cu128 max_fusion_size 64 mismatches 0 of 28800 max_abs 0   (default, fused)
torch 2.7.1+cu128 max_fusion_size 1  mismatches 0 of 28800 max_abs 0   (NOFUSE=1 control)
```

Zero mismatches with the fusion ALLOWED (`max_fusion_size` defaults to 64 in
2.7.1, not disabled) - i.e. Inductor no longer emits the impossible
`tmp4 & tmp36` (`row<100 AND 100<=row<200`, always-false) load mask that
corrupted `op195+op207`'s fused kernel on 2.5.1. This is the single
highest-value result in this evaluation: **the miscompile this project
works around with a `torch.library.custom_op` fusion barrier
(`salad_torch/modules/cond_prologue.py:_p7_barrier`) does not reproduce on
2.7.1's Inductor.** I did not bisect which of the many Inductor releases
between 2.5.1 and 2.7.1 fixed it (out of scope, and PyTorch's public
changelogs don't index individual scheduler-fusion fixes finely enough to
find it without a bisection build matrix) - only that it is gone at the
tested endpoint.

This does **not** by itself prove the barrier is safe to remove: the barrier
protects against a *class* of fusion decision, and "gone in this one
standalone repro" is not "provably absent from every shape this model
compiles." A real removal would need to re-run this reproducer's exact
scheduler trace inside the full model (as N031-p7-fusion-rootcause did for
2.5.1) before deleting the barrier, not just point at this standalone
result. Recommendation in section 7 accounts for this.

## 3. Does the whole port gate still pass? Mostly - one narrow numeric miss

`salad_jobs/n033_gate.sbatch`, job 908348, against a `git archive
4cb8ef4` snapshot in `$TMPDIR` (HEAD at task start; never the live tree,
which a concurrent sibling agent churned mid-task - see section 8):

| check | result |
|---|---|
| `test_model_parity` default_vp | **PASS** (11/11, coord dev max 0.00001 A) |
| `test_model_parity` default_vp_scaled | **PASS** (11/11, coord dev max 0.00001 A) |
| `test_model_parity` default_ve_scaled | **FAIL** - only `atom_pos`: rel=2.989e-05 vs the fixed 2e-05 tolerance (abs 2.17e-03); all other 10 outputs incl. `pos`/`trajectory`/`aatype` PASS at comparable magnitudes |
| `test_model_parity` multimotif_vp | **PASS** (11/11, coord dev max 0.00015 A) |
| `test_geometry_parity` | **PASS** 38/38 |
| `test_stack_parity` | **PASS** |
| `test_encoder_parity` | **PASS** (3/3 length x chain combos) |
| `test_sampling_parity` | **PASS** (worst/floor 7.0x, final/floor 8.4x, both under the test's own threshold) |
| `test_design_quality.py --n 160 --preset fast` | **PASS - 0/160 diverged (0.0%, limit 10%)**, CA-CA 3.7609 A, Rg 13.00 (native ~12-13) |

Net: **8/9 checks pass outright; the one failure is a single narrowly-missed
tolerance on one of four configs' `atom_pos`, not a functional break, not a
divergence, and not visible in the metric that actually gates ships
(design-quality n=160, 0/160 diverged - identical to the 2.5.1 baseline).**
`atom_pos` is *built from* `angles`, which itself passes (rel=8.931e-04,
right at its own 1e-3 budget) - consistent with ordinary fp32
summation-order drift from a different Inductor codegen on this one config
class, the same class of difference `test_model_parity.py`'s own docstring
already documents for `raw_angles` ("fusing the pair-feature linears...moved
it by ~2e-5 relative, expected fp32 behaviour, not a regression"). Only
`default_ve_scaled` is affected; the structurally similar `default_vp_scaled`
is clean. Not root-caused further (would need a per-config Inductor IR diff,
out of scope for an evaluation) - reported as a real, narrow finding, not
waved away.

## 4. Speed: `fast` is faster, `exact` is flat

`salad_jobs/n033_speed.sbatch`, job 908349, lean step at L=100, 3 reps, one
fresh process per rep (same methodology as N014's `bench_p7.py`):

| preset | 2.5.1+cu121 (current, committed) | 2.7.1+cu128 (this eval) | delta |
|---|---|---|---|
| exact | 3.087 ms | 3.090 / 3.090 / 3.091 ms (mean 3.090) | **+0.1%, flat** |
| fast | ~2.54 ms | 2.153 / 2.155 / 2.149 ms (mean 2.152) | **-15.3%, faster** |

Both presets are reproducible to +-0.002 ms across the 3 reps (tight). Cold
compile time also dropped: 192-200s here vs the brief's cited ~310s on
2.5.1 (measured independently again in section 5's Mega-Cache `cold` phase:
251s at L=100/`fast`, in the same ballpark - the two numbers differ because
`x_bench_n033.py` and `x_megacache.py` warm up slightly different graph
sets, not a discrepancy worth chasing further).

Read: 2.7.1's Inductor fuses the `fast` preset's `decompose_small_bmm`-heavy
graph measurably better than 2.5.1's did (or the P7 fusion-barrier custom-op
interacts better with the newer scheduler now that the underlying miscompile
class is fixed - not disentangled, both plausible, not distinguished here).
`exact` (which enables neither `decompose_small_bmm` nor
`max_autotune_gemm`) sees no change, consistent with the win being concentrated
in exactly the machinery `fast` turns on.

### Rejected-optimization recheck: matmul-template autotune instability REPRODUCES

`salad_jobs/n033_det.sbatch`, job 908350: reused N030's own
`x_common.py`/`x_arm.py`/`x_dump.py`/`x_cmp.py` **completely unmodified**
(already `$SALAD_TREE`-parameterized) to redo N020/N030's cross-compile
bitwise check - 3 independent COLD-cache compiles of `forward_core`,
`SALAD_OPT_DETSEG=1` (removes the model's only other GPU nondeterminism
source, the segment-reduction atomicAdd), L=100:

| arm | what it is | rep1 vs rep2 `pos` max\|d\| | rep1 vs rep3 `pos` max\|d\| |
|---|---|---|---|
| `off` | shipped reference (no autotune templates) | 1.359e-05 | 0.0 (bit-identical) |
| `all` | `max_autotune_gemm` unfiltered - N020's rejected -14% arm | **1.376e-04** | **1.521e-04** |
| `decomp32` | `decompose_mm_pass` - what's actually SHIPPED in `fast` (not autotune-based) | 2.098e-05 | 0.0 (bit-identical) |

`off`'s own cross-compile noise floor (1.36e-05, matching N020's originally
reported 3.5e-05 on 2.5.1 in order of magnitude - ordinary Inductor codegen
variance, present on both torch versions) is the yardstick. `decomp32` sits
AT that floor (2.10e-05), matching N030's finding that it is "perfectly
reproducible across cold compiles" - **reconfirmed on 2.7.1, the shipped
mechanism is still safe from this angle.** `all` sits roughly **10x above**
the floor (1.4-1.5e-04), the same order of magnitude as N020's originally
measured 8.87e-05 on 2.5.1+cu121. **Conclusion: torch 2.7.1 does NOT fix the
matmul-template autotune timing race; N020/N030's rejection of `max_autotune_gemm`
still holds on the new torch, unchanged in kind or magnitude.**

## 5. Mega-Cache: the API is real and the artifact is honored, with one process-ordering footgun

`torch.compiler.save_cache_artifacts`/`load_cache_artifacts` exist in 2.7.1
(absent in 2.5.1, confirmed via `hasattr`, job 908347). Three-phase test
(`x_megacache.py`, `cold`/`save`/`load`, each its own FRESH process with its
own EMPTY on-disk `TORCHINDUCTOR_CACHE_DIR` so no phase's compile time is
served by Inductor's ordinary on-disk FX cache):

**First attempt (job 908347) hit two bugs, both in my harness:**
1. `load_cache_artifacts(blob)` called as the very first thing to touch
   `torch` in a fresh process crashed:
   `ImportError: cannot import name 'CacheArtifactManager' from partially
   initialized module 'torch.compiler._cache' (most likely due to a circular
   import)`. Mechanism: `torch.compiler.__init__` imports `._cache`, whose
   own import chain (`_inductor.remote_cache` -> `_dynamo.utils` ->
   `_dynamo.__init__` -> ... -> `_dynamo.pgo`) re-imports
   `torch.compiler._cache` while it is still mid-import - a genuine circular
   import inside torch 2.7.1 itself, but one that only bites when
   `load_cache_artifacts` is the FIRST thing to pull in the full
   `torch._dynamo` package. **This is a real, reportable torch 2.7.1
   footgun** (worth a short upstream note if this project ever adopts
   Mega-Cache), fixed on my side with one line: `import torch._dynamo` before
   calling `load_cache_artifacts`.
2. My own missing `torch.manual_seed(0)` meant `FastDenoiser`'s internal
   (caller-unseeded) neighbour/uniform draws differed across the three
   independent processes, so the `cold` vs `save` output comparison showed
   `bit_identical=False` for reasons that had nothing to do with the cache
   (ordinary cross-process RNG, not a Mega-Cache correctness issue). Fixed
   by seeding globally and rerunning.

**Rerun (`n033_megacache2.sbatch`, job 908363) with both fixes plus
`SALAD_OPT_DETSEG=1`:**

| phase | compile_time | notes |
|---|---|---|
| `cold` | 251.74 s | fresh empty cache dir, nothing preloaded |
| `save` | 252.68 s | same cost as `cold` (expected - saving happens AFTER compiling), then serializes 108,490,902 bytes (~103 MiB: 3 `inductor_artifacts`, 189 `autotune_artifacts` `.best_config` files - Mega-Cache captures autotuning decisions too, not just codegen - 3 `aot_autograd_artifacts`, 0 `pgo_artifacts`) |
| `load` | **25.79 s** | DIFFERENT fresh empty cache dir, `load_cache_artifacts()` first (0.139 s to deserialize) |

**Mega-Cache cuts compile time from 251.7s to 25.8s at L=100/`fast` - a 9.8x
reduction (90% removed)**, matching the brief's own cited cold/warm figures
(~310s / ~68s) in direction and putting a real number on this project's
specific model/preset. The load-order fix (section headline) held with
`SALAD_OPT_DETSEG=1` added too - no repeat of the circular-import crash.

Bitwise output equality is a **wash, not a pass**: `cold` vs `save` (both
independently cold-compiled - NO Mega-Cache artifact involved in either)
already differ by max\|d\|=13.59 across all 15 output tensors; `save` vs
`load` (the actual Mega-Cache comparison) differ by max\|d\|=17.14 - the
SAME order of magnitude as the no-cache control, not a step up. Both are far
above the ~1e-5 A noise floor the determinism recheck (section 4b) measured
for `forward_core` alone under `SALAD_OPT_DETSEG=1` with fixed inputs and no
sampler-level randomness. The difference is `FastDenoiser.warmup`'s own
internal neighbour/Gumbel draws, which come from the GLOBAL default RNG at a
point in each process's history that depends on how much of that RNG's
stream earlier setup consumed (model construction order, whether the `load`
branch touched `torch._dynamo` first, etc.) - `torch.manual_seed(0)` only
guarantees reproducibility from the moment it is called if every process
consumes the SAME NUMBER of draws before and after that point, which this
harness does not enforce (this project's own parity fixtures dodge exactly
this by replaying a **fixed uniform field** instead of trusting a global
seed - see `test_model_parity.py`'s docstring). **Because the no-cache
control (`cold` vs `save`) is already just as far off as the
with-cache comparison (`save` vs `load`), the large max\|d\| is attributable
to this harness's own unpinned sampler randomness, not to Mega-Cache** - but
this run does not POSITIVELY prove bit-exactness either, only that Mega-Cache
does not make cross-process agreement any worse than it already was. A truly
clean bit-exactness check would need the fixed-uniform-field technique this
project already uses elsewhere for parity fixtures, which this evaluation
did not have time to retrofit into the Mega-Cache harness.

## 6. sm_120 / Blackwell: capability confirmed (not measured - no such GPU here)

Job 908347, on an actual H100 (sm_90) via `--nv`:

```
torch.version.cuda        = 12.8
torch.cuda.get_arch_list() = ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120']
sm_120 in arch_list: True
```

This is a **capability claim**: the installed torch 2.7.1+cu128 wheel was
built with `sm_120`/`compute_120` in its kernel image and CUDA 12.8 is new
enough to target Blackwell, so the toolchain COULD run on a real sm_120 card
- no Blackwell GPU exists on this cluster to actually run a kernel on one.

## 7. Recommendation: UPGRADE, but stage it - do not flip the default env yet

**Upgrade later, in a staged way** - not "now" as a same-day env swap, not
"never." Evidence for each:

**For upgrading:**
- The #1 blocker (P7 fusion miscompile, the reason for a custom-op fusion
  barrier at all) reproduces as FIXED at the standalone-reproducer level
  (section 2) - the strongest single motivator in the brief.
- The full gate passes except one narrow, non-quality-affecting tolerance
  miss on one of four configs (section 3) - and the metric that actually
  gates ships, `test_design_quality.py --n 160`, is 0/160 diverged, matching
  the 2.5.1 baseline exactly.
- `fast` is 15.3% FASTER at L=100 with zero code changes (section 4) - pure
  upside, no tradeoff identified.
- Mega-Cache is real and delivers a measured **9.8x compile-time cut**
  (251.7s -> 25.8s at L=100/`fast`) once one process-ordering footgun is
  worked around (section 5) - genuinely removes the "ship a cold compile to
  every user" problem the brief names as blocker #2. (Bitwise output equality
  was a wash, not a clean pass, for a harness reason unrelated to Mega-Cache
  itself - see section 5's caveat before citing this as "proven bit-exact".)
- sm_120 is confirmed targetable (section 6) - blocker #3 resolved as a
  capability, satisfying the project's explicit consumer-Blackwell
  requirement.
- The REJECTED matmul-template optimization's instability is UNCHANGED
  (section 4b) - reassuring in the sense that nothing about the upgrade
  silently destabilizes a decision this project already made carefully; it
  is not itself a reason to upgrade, but rules out "maybe 2.7 also quietly
  fixes this and we should revisit" as a distraction.

**Against doing it same-day, as a same-day default-env flip:**
- One real (if narrow) parity regression exists (`default_ve_scaled`
  `atom_pos`, section 3) that was not present on 2.5.1 and has not been
  root-caused - it needs either a wider tolerance (if genuinely just fp32
  reordering) or an actual fix, not a shrug, before this config ships against
  the new torch.
- The private APIs the brief specifically flagged
  (`torch._functorch.compile_utils.fx_graph_cse`,
  `torch._inductor.config.{joint_custom_post_pass,max_autotune_gemm,
  decompose_mm_pass,aggressive_fusion}`) were NOT individually exercised
  here beyond what the gate and the determinism recheck happen to touch
  (`x_arm.py`'s `apply("all")`/`apply("decomp32")` do exercise
  `max_autotune_gemm` and `decompose_mm_pass` directly, and both worked; the
  gate's own `fast` preset run exercises `fx_graph_cse` via
  `joint_custom_post_pass` and `aggressive_fusion` indirectly, and also
  worked). `gemm_exact.py` - the module with the heaviest private-API
  surface (`torch._inductor.kernel.{mm,bmm}`, `AlgorithmSelectorCache`,
  `TritonTemplateCaller`) - was never invoked in this evaluation (`x_arm.py`
  only calls `gemm_exact.uninstall()` for the arms tested here, never
  `.install()`), since it is a research/debug knob that was never shipped
  (N030's Outcome: "moot for exact/deterministic"). If this project ever
  wants to revive it, that revival needs its own compatibility pass - not
  covered by "the gate passed."
- The custom-op fusion barrier's removal needs its own targeted test inside
  the full model (the standalone reproducer being fixed is necessary
  evidence, not sufficient proof for every shape this model compiles) -
  section 2's caveat.
- This container is genuinely minimal (numpy+torch only) precisely because
  the gate tests don't need more - but the moment this env is used for
  anything beyond running these tests (e.g. `--template`'s `Bio.PDB`, or
  regenerating fixtures via the JAX reference code), it is missing
  dependencies on purpose. Fine for an evaluation; not yet a drop-in
  replacement for `triad`.
- Only ONE torch 2.7.x point release and ONE CUDA variant (cu128) were
  tested. PyPI shows stable releases through **2.14.0 (released
  2026-09-02, 25 days before this evaluation)** - torch's own release cadence
  has moved on five more minor versions since 2.7. This evaluation answers
  "is >=2.7 enough," which is what the brief asked; it does not by itself
  make the case for 2.7.1 specifically over a more current stable release,
  which would carry its own (likely smaller, given the trend visible here)
  set of private-API breaks and possibly an already-fixed
  `default_ve_scaled` tolerance.

**Staged path, if/when this project decides to move:**
1. Root-cause (not just tolerance-widen) the `default_ve_scaled` `atom_pos`
   miss before shipping that config against 2.7+.
2. Re-run this same reproducer-plus-gate-plus-speed battery against a more
   current stable release (2.9.1 or later; 2.14.0 if PyPI's index is to be
   trusted at 25-days-old) before committing to a specific target version -
   the gap between 2.5.1 and "current" is now large enough that picking the
   nearest satisfying version (2.7) rather than the best available one is
   leaving improvements on the table for free, per this item's own evidence
   that things got faster and more correct between 2.5.1 and 2.7.1.
3. Re-run the standalone P7 reproducer against the full model (not just the
   isolated `minimal_conditional.py`) before deleting the fusion barrier -
   confirm inside `cond_prologue.py`'s real graph, not just the extracted
   repro.
4. Build the new env as a SECOND apptainer `.sif` (as done here), run the
   whole gate + a full design-quality n=160 pass through the actual shipped
   `generate.py` entry point (this evaluation used the harness scripts, not
   the CLI, for speed/determinism measurements - the gate step DID use
   `test_design_quality.py`'s own call into `generate.py`, so that path is
   covered) before ever touching the `triad` env definition.
5. Only then, cut the `triad` env over (new conda env or a wholesale
   apptainer-based replacement of the current bare-conda job pattern -
   worth reconsidering given how cleanly the container route worked here)
   and update every `salad_jobs/*.sbatch` script's `PATH=.../envs/triad/bin`
   line in one commit, gated on this project's own before/after suite.

## 8. A note on shared-tree hazards (not a torch finding, but load-bearing for how this was run)

Mid-task, this repo's single working directory (no separate worktree was
used for this evaluation - by design, since I never touched live source,
only `git archive`d snapshots) was switched by a concurrent process (almost
certainly sibling agent N032/N028, who has since committed
`18d30d4 N028: land O8...` and `9ab8fac docs: rebuild the README...`) from
`n006-manual-cudagraph` to a `release` branch and back again, and my
in-progress `.agent-research/results/N033.../` files were swept into that
commit (their content was not touched, and every path in this results dir
was intact after `git checkout n006-manual-cudagraph` restored the branch).
This is exactly the class of hazard the task's "work from a git-archive
snapshot, never the live tree" rule anticipates - it protected all of this
item's actual GPU evidence (every job snapshots its own pinned commit,
`4cb8ef4`, into `$TMPDIR` before running anything) but NOT this notebook
item's own in-progress edit, since notebook records necessarily live in the
shared tree. No evidence here was compromised; flagged so the next person
touching this shared checkout knows branch switches happened.
