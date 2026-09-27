# N033 · Torch upgrade evaluation: 2.5.1+cu121 -> >=2.7 (miscompile, mega-cache, sm_120)

type: task · date: 2026-09-27 · status: done

Brief called this "N031" (`.agent-research/results/N031-torch-upgrade/`); N031
was already allocated to the sibling "P7 fusion miscompile" item
(`N031-p7-fusion-rootcause`, itself a duplicate of the earlier `N026-p7-fusion-rootcause`)
by the time this item was created via `new-item.sh`, which returned N032 -
also pre-claimed by the brief for a concurrent sibling agent's Triton-fusion
work. Bumped to N033 by hand (rename + INDEX row edit) to avoid a third
collision, same failure mode as N027/N030's postmortems.

## Goal

Evaluate upgrading the project's default torch (2.5.1+cu121, the `triad`
conda env - NOT to be modified, shared by other live projects and every job
in this repo) to >=2.7, which the brief frames as unlocking three things:
(1) does a newer inductor fix the op195+op207 fusion miscompile
(`N031-p7-fusion-rootcause`/`N026-p7-fusion-rootcause`, impossible AND-of-disjoint-range
load mask, 23,070/28,800 wrong relpos bins) that the shipped code currently
works around with a `torch.library.custom_op` fusion barrier; (2) Mega-Cache
(`save_cache_artifacts`/`load_cache_artifacts`, absent in 2.5.1) to ship a
prebuilt inductor cache and remove the ~310s cold compile; (3) sm_120
(Blackwell) targeting, impossible on cu121. Recommend upgrade now / later /
not, with evidence; if upgrade, list exactly what code has to change (several
PRIVATE inductor/Dynamo APIs in `gemm_exact.py`/`fast.py`/`opt.py` are named
in the brief as likely breakage). Evaluation only - no code changes to the
port, no changes to the `triad` env.

## Plan

1. Build a torch>=2.7 env WITHOUT touching `triad` or blowing an inode quota.
   AST-scanned `salad_torch/tests/scripts` first: the gate tests import only
   `numpy`+`torch` at runtime (jax/haiku/Bio/biotite/einops/scipy/dm-tree are
   `triad`-env leftovers for the JAX reference code / optional `--template`
   path, unused by the port's own tests) - so the new env can be minimal.
   Apptainer .sif, built with `--fakeroot` in node-local `/tmp` on a login
   node (has internet; compute nodes don't), copied as ONE file to
   `/hnvme/workspace/v181bb10-salad/` - same pattern as BindCraft/ORCA.
2. Run the standalone no-salad reproducer (`minimal_conditional.py`) fused
   vs `NOFUSE=1` under the new torch.
3. Run the full gate (parity x4 configs, geometry/stack/encoder/sampling,
   `test_design_quality.py --n 160 --preset fast`) against a `git archive
   <pinned commit>` snapshot in `$TMPDIR` (never the live tree - N032 may be
   editing it), copying gitignored `fixtures/*.npz` and symlinking
   `params`/`ae_params` into the snapshot by hand (none of the three are
   tracked by git, confirmed via `git ls-files`).
4. Re-measure lean step L=100, exact/fast, 3 reps, fresh process per rep
   (N014's own methodology - compiling several shapes in one process
   degrades cudagraphs). Baseline: exact 3.087 ms, fast ~2.54 ms.
   Also re-run N020/N030's REJECTED matmul-template autotune determinism
   check (3 cold compiles, bitwise compare) by reusing N030's own
   `x_common.py`/`x_arm.py`/`x_dump.py`/`x_cmp.py` unmodified (already
   resolve the tree via `$SALAD_TREE`) - arm `all` (`max_autotune_gemm`
   unfiltered, N020's rejected -14%) vs `off` (shipped reference) vs
   `decomp32` (what's actually shipped in `fast`, not autotune-based, a
   determinism control).
5. Mega-Cache: 3 processes, one node, cold/save/load, each its own EMPTY
   on-disk `TORCHINDUCTOR_CACHE_DIR` so `save`/`load`'s compile times are
   genuine and not served by inductor's normal on-disk cache; bitwise-compare
   outputs.
6. sm_120: `torch.cuda.get_arch_list()` + CUDA version only - no Blackwell
   card here, capability claim not a measurement.

## Changes

None to the port. New, evaluation-only artifacts in this results dir:
`torch27_cu128.def` (apptainer recipe), `x_bench_n033.py` (ROOT
parameterized via `$SALAD_TREE`, else identical to N014's `bench_p7.py`),
`x_megacache.py` + `x_cmp_megacache.py` (new), and unmodified copies of
N030's `x_common.py`/`x_arm.py`/`x_dump.py`/`x_cmp.py` (already
`$SALAD_TREE`-parameterized, reused as-is) plus `minimal_conditional.py`
(unmodified copy of the standalone reproducer). Job scripts
`salad_jobs/n033_{quick,gate,speed,det}.sbatch`.

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

Environment: built `torch27_cu128.sif` (python:3.11-slim-bookworm +
torch==2.7.1+cu128 + numpy==1.26.4 only), `apptainer build --fakeroot` in
node-local `/tmp` on `helma3` (login node, confirmed internet access),
65s wall. Inode cost: 0 on `/home/hpc` (36k headroom), 0 permanent on
node-local `/tmp` (ephemeral build scratch, not inode-capped, cleaned up
after), **+1 inode on `/hnvme`** for the 3.7 GB `.sif` copied to
`/hnvme/workspace/v181bb10-salad/torch27_cu128.sif` (hnvme df -i: 9,601,541 /
37,131,382,848 used, i.e. 0.03% - the inode pressure the brief warns about is
a project-level soft quota on FILE COUNT, not this raw mount, and a single
extra file is negligible either way). Sanity: CPU-only smoke (login node, no
GPU) - plain `import salad_torch.modules.{model,fast,gemm_exact,opt}` OK
under 2.7.1 (all private-API imports in those files are lazy, inside
functions, not at module top level); `tests/test_geometry_parity.py` 38/38
PASS on CPU under 2.7.1 with the pinned-commit snapshot.

First batch (908271/908272/908273/908274) all failed within 2-9 minutes:
`python:3.11-slim-bookworm` has NO C compiler at all, and Triton JIT-builds a
tiny CPython extension (`driver.c`) on its first kernel launch even though
Inductor itself only emits Triton/C++ source text - "Failed to find C
compiler" on every single `torch.compile` call. Fixed by adding
`build-essential` to the def file and rebuilding (73s). `n033_det.sbatch`
also had its own bug (forgot `SALAD_TREE=$SNAP` on the `x_cmp.py` calls -
`KeyError: 'SALAD_TREE'`), fixed and resubmitted alongside.

Second batch, GPU (H100, one job per configuration): 908347 (quick: capability
+ reproducer + Mega-Cache attempt 1), 908348 (gate), 908349 (speed), 908350
(det) - all COMPLETED, realistic runtimes (2-24 min). Mega-Cache's first
attempt (908347) hit two more bugs, both mine: `load_cache_artifacts()`
called before anything touched `torch._dynamo` crashed with a genuine
torch-2.7.1 circular import (`torch.compiler._cache` partially initialized);
and a missing `torch.manual_seed(0)` confounded the output-equality check
with ordinary cross-process sampler randomness. Fixed both, reran as
908363 (`n033_megacache2.sbatch`).

Total GPU time across both batches: ~1h wall on 1x H100 per job, well under
the 25 GPU-h approval threshold (nowhere close - this whole evaluation used
single-digit GPU-hours).

Full numbers, per deliverable, in RESULTS.md. Headlines:
- **Reproducer: FIXED.** `minimal_conditional.py` gives 0/28800 mismatches on
  2.7.1 (was 22,222-23,070/28,800 wrong on 2.5.1).
- **Gate: 8/9 pass.** One narrow tolerance miss (`default_ve_scaled`'s
  `atom_pos`, rel=2.989e-05 vs a 2e-05 floor) on ONE of four parity configs;
  everything else including `test_design_quality.py --n 160 --preset fast`
  (0/160 diverged, matching the 2.5.1 baseline exactly) passes clean.
- **Speed: `fast` -15.3% (2.54ms -> 2.152ms), `exact` flat (+0.1%).** Cold
  compile time also dropped (~310s -> ~200-250s).
- **Rejected matmul-template optimization: UNCHANGED.** Its cross-compile
  instability reproduces at the same ~10x-above-noise-floor magnitude on
  2.7.1 as N020 measured on 2.5.1 (1.4-1.5e-04 A vs a 1.36e-05 A baseline
  noise floor); the shipped `decompose_small_bmm` mechanism stays AT the
  noise floor on both torch versions.
- **Mega-Cache: works, 9.8x compile-time cut** (251.7s -> 25.8s), after
  fixing the load-order bug above. Bitwise output-equality check was a wash
  (cold-vs-save, no cache involved, already differs by the same amount as
  save-vs-load) - not proof of bit-exactness, but no evidence Mega-Cache
  makes cross-process agreement any worse either; the confound is this
  harness's unpinned sampler-level randomness, not Mega-Cache.
- **sm_120: confirmed targetable** (`torch.cuda.get_arch_list()` includes
  `sm_120`/`compute_120`, CUDA 12.8) - capability claim, no Blackwell card
  here to measure on.

## Outcome

**Recommendation: upgrade later, staged - not a same-day default-env flip,
not "never."** Full reasoning with evidence in RESULTS.md section 7. Short
version: the #1 blocker (miscompile) is fixed, the gate is essentially clean,
`fast` got faster for free, Mega-Cache delivers a real 9.8x compile-time cut,
and sm_120 is confirmed targetable - all strong "yes, eventually" signals.
Against doing it today: one unexplained (if narrow) new parity miss needs
root-causing first, only ONE point release (2.7.1) and one CUDA variant
(cu128) were tested while PyPI already shows stable releases through 2.14.0
(2026-09-02) - so picking 2.7 specifically over "whatever is current" would
be leaving free improvements on the table by this item's own evidence trend
- and the custom-op fusion barrier's removal needs its own in-model
(not just standalone-reproducer) confirmation before deletion.

Also recorded: mid-task, a concurrent process (sibling agent, since
committed as `18d30d4`/`9ab8fac`) switched this shared working directory's
branch away from `n006-manual-cudagraph` and back, sweeping this item's
in-progress files into its own commit along the way. Content was unaffected
(verified after `git checkout n006-manual-cudagraph` restored it) and no GPU
evidence was at risk (every job snapshots its own pinned commit via `git
archive` into `$TMPDIR`, never reads the live tree) - but it is the exact
hazard class the snapshot rule anticipates, just hitting the notebook record
instead of the port code. See RESULTS.md section 8.

## Outputs
<!-- files in .agent-research/results/N033-slug/ + external absolute paths -->

* `.agent-research/results/N033-torch-upgrade-evaluation-2-5-1-cu121-2-7/` -
  `RESULTS.md` (full writeup, all 8 sections), `torch27_cu128.def`, harness
  scripts (`x_bench_n033.py`, `x_megacache.py`, `x_cmp_megacache.py`, and
  unmodified copies of N030's `x_common.py`/`x_arm.py`/`x_dump.py`/`x_cmp.py`
  plus the standalone `minimal_conditional.py`), `logs/` (all 6 job logs,
  including the first failed attempt for the record).
* `/hnvme/workspace/v181bb10-salad/torch27_cu128.sif` - the built environment
  (3.8 GB, evaluation only, NOT wired into any default job path; `triad`
  conda env on this same repo is untouched).
* `salad_jobs/n033_{quick,gate,speed,det,megacache2}.sbatch` - job scripts
  (hard rule: job scripts live under `salad_jobs/`, not the results dir).

## Next

* Root-cause `default_ve_scaled`'s `atom_pos` tolerance miss (RESULTS.md
  section 3) before any config ships against torch>=2.7.
* Re-run this same battery against a more current stable release (2.9.1+,
  possibly 2.14.0) before picking a specific upgrade target - the gap
  between 2.5.1 and "current" (as of 2026-09-27) is now 9 minor versions,
  and 2.7 was chosen here only because the brief itself framed the ask
  around it (cu128/sm_120), not because it is necessarily the best target.
* Re-run the P7 reproducer INSIDE the full model (not just
  `minimal_conditional.py` standalone) before deleting the fusion barrier.
* If upgrading, build the replacement `triad`-equivalent env with the deps
  this evaluation's minimal image deliberately left out (`Bio`/biotite/etc.
  for `--template` and fixture regeneration) - this container is
  evaluation-only, not a drop-in replacement as built.
