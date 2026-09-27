# N033 · Torch upgrade evaluation: 2.5.1+cu121 -> >=2.7 (miscompile, mega-cache, sm_120)

type: task · date: 2026-09-27 · status: open

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

GPU jobs submitted (H100, one job per configuration, `n033_{quick,gate,speed,det}.sbatch`):
jobid TBD - see RESULTS.md once complete.

## Outcome

TBD - jobs running.

## Outputs
<!-- files in .agent-research/results/N033-slug/ + external absolute paths -->

* `.agent-research/results/N033-torch-upgrade-evaluation-2-5-1-cu121-2-7/` -
  def file, harness scripts, job logs (copied from `salad_jobs/`), RESULTS.md.
* `/hnvme/workspace/v181bb10-salad/torch27_cu128.sif` - the built environment
  (evaluation only, NOT wired into any default job path).
* `salad_jobs/n033_{quick,gate,speed,det}.sbatch` - job scripts (hard rule:
  job scripts live under `salad_jobs/`, not the results dir).

## Next

TBD once GPU jobs return.
