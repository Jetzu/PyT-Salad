# N036 · Symmetric generation's 8 extra flags (upstream eval_sym.py)

type: task · date: 2026-09-27 · status: done

## Goal

Close the gap between this port's already-shipped `--sym` (screw/cyclic
symmetry, ported from `eval_noise_schedule_benchmark.py` per
`notes/specs/06_generation_flags.md` §1) and upstream's SEPARATE, dedicated
symmetric-generation script, `salad/training/eval_sym.py`, which has 8 more
flags that script never had a home for: `--fixcenter_threshold`,
`--compact_lr`, `--clash_lr`, `--f_small`, `--f_strand`, `--mix_output`,
`--mode`, `--sym_noise`. Third of three parallel gap-closing ports into
`scripts/generate.py` (siblings: shape conditioning `--blobs` N034,
multi-state design N035) - keep the diff minimal/localised, per the brief.

## Plan

No local upstream copy existed; fetched directly:
`curl -sfL https://raw.githubusercontent.com/mjendrusch/salad/master/salad/training/eval_sym.py`
(the filename in the brief, `eval_ve_sym.py`, 404s - the real name is
`eval_sym.py`, confirmed via the GitHub contents API) plus
`salad/inference/symmetry.py`, `salad/inference/sampling.py`,
`salad/inference/model.py`, `salad/training/eval_repeat.py`. Read all of them
in full before writing anything (`.agent-research/results/N036-symmetry-flags/upstream/`
has the two files the flags actually come from).

Worked out from source, not the README, exactly what each of the 8 flags
does - full reasoning in `salad_torch/inference/symmetry.py`'s module
docstring (the "--- N036 ---" section). Headline findings:

* `--mode rotation` is **dead code upstream, not merely undesirable**:
  `eval_sym.py`'s `model_step` always calls
  `sym_op.replicate_pos(mix_output(pos, do_radius), do_radius)` (2 positional
  args after `first`), but `ScrewVP.replicate_pos(self, first)` (the
  `--mode rotation` class) doesn't declare `do_radius` at all - unlike its
  own `couple_pos`/`select_pos`, which accept-but-ignore it. Every
  `--mode rotation` invocation of `eval_sym.py`, in every released version,
  raises `TypeError` inside JAX's trace on step 0. Verified standalone with
  no jax dependency at all (a pure Python arg-count check, see
  `evidence/prove_screwvp_bug.py`) - reproduced the exact `TypeError`.
  Ported as a hard, immediate `SystemExit` citing this, not a
  reimplementation (no working upstream referent exists to match).
* `--sym_noise` was **not** already unconditional in the way the brief
  hypothesised it might be for the OTHER symmetry mechanism - the port's
  existing hook B (`screw_replicate` on `pos_noised`) runs unconditionally
  whenever `--sym` is active, matching upstream's own *default*
  (`sym_noise="True"`), but upstream's `eval_sym.py` genuinely gates it
  (`config.replicate_noise`) and this port had no way to turn it off. Wired
  as a real gate around the existing hook B call.
* `config.symm` / the class-based `Screw` in `salad/inference/symmetry.py`:
  reconfirmed dead (already established by spec 06 for the OTHER script);
  none of the 8 flags route through it - `eval_sym.py` has its own,
  textually-duplicated `Screw`/`ScrewVP` classes and never imports
  `salad/inference/symmetry.py` at all.
* `--fixcenter_threshold`, `--compact_lr`, `--clash_lr`: all three key off
  the CURRENT step's *scaled* noise level (`data["t_pos"][0]`, i.e. after
  `--timescale_pos`), not the raw step fraction `--sym_threshold` already
  uses - ported by recomputing `timescale_pos_fn(raw_t)` in the CLI (pure,
  cached, semantically free) rather than threading a new value through
  `sampling.py`.
* `--mix_output`: the SAME couple/select choice `--sym_mean` already
  controls, upstream's OTHER script's name/values for it (that script never
  had `--sym_mean`). Folded into `mean` at parse time rather than
  reimplemented, since it is the identical mechanism, not a second one.
* `--f_small`/`--f_strand`: a CLI-level reject-and-regenerate design filter
  (ALA/GLY fraction, DSSP strand fraction), not model geometry - upstream
  retries unboundedly, this port caps it at 20 attempts (both flags default
  to 1.0, an exact no-op, so the cap is inert unless a caller opts in).

## Changes

* `salad_torch/inference/symmetry.py`: `SymmetrySpec` + `parse_sym` extended
  (8 new fields/params, `mode="rotation"` raises `SystemExit`); `screw_replicate`
  gained `do_radius: bool = True` (existing call sites unchanged, default
  preserves current behaviour exactly); new `compact_clash_update` function
  (a THIRD, distinct clash formula from the existing `clash_repulsion_grad` -
  see its docstring). Module docstring rewritten with the full per-flag
  reasoning.
* `scripts/generate.py`: 8 new argparse flags in the existing `sym` group;
  `sym_filter_check` (new, `--f_small`/`--f_strand`); hook A/B/C extended for
  `do_radius`/`compact_lr`/`clash_lr`/`sym_noise`; a reject-and-regenerate
  retry loop around the per-design `sample()` call.
* `.agent-research/results/N036-symmetry-flags/`: upstream sources fetched,
  `evidence/n036_evidence.py` (24 checks: unit-level + full-CLI, each flag's
  measurable effect), `evidence/prove_screwvp_bug.py` (the `--mode rotation`
  reproduction), `reconstruction/` (see below).

**Shared-tree note**: two sibling agents (N034 shape-conditioning, N035
multi-state) were editing `scripts/generate.py`/`salad_torch/inference/__init__.py`
concurrently, both already mid-flight (their own INDEX rows still `open`,
their staged/unstaged diffs present in the working tree throughout this
task). `salad_torch/inference/symmetry.py` is entirely mine (no sibling
touches it - confirmed via `git status`, plain `M` not `MM`, no other agent's
markers in its diff). `salad_torch/inference/__init__.py` needed NO changes
at all: `compact_clash_update` is imported directly from
`salad_torch.inference.symmetry` in `scripts/generate.py`, the same pattern
the existing `parse_sym`/`screw_replicate` already use - the package
`__init__.py`'s `from .symmetry import ...` line is byte-identical between
HEAD and the current working tree.

`scripts/generate.py` itself DOES need the same surgery N035 documented:
reconstructed "HEAD + only my hunks" (`reconstruction/build_mine.py`, a
deterministic sequence of anchored string replacements against
`git show HEAD:scripts/generate.py`, each anchor chosen to land on
HEAD-original text, not sibling-authored text) and verified it two ways
before touching the shared index: (1) `py_compile`, (2) extracted a
COMPLETELY ISOLATED pristine-HEAD copy via `git archive HEAD | tar -x -C
/dev/shm/...` (never touching the shared working tree), dropped the
reconstruction in as `scripts/generate.py`, and re-ran the full 24-check
evidence suite against THAT tree end to end - all 24 pass, confirming the
reconstruction is not just textually plausible but functionally
self-sufficient (does not depend on anything N034/N035 added). The patch
(`reconstruction/generate.py.mine.patch`, HEAD -> HEAD+mine, 287 lines) is
regenerated fresh against whatever HEAD is current immediately before the
actual `git apply --cached` + commit, in case a sibling has landed by then.

## Evidence

GPU job 908569 (h100), CPU job 908570 (cpu partition) - full parity +
`test_design_quality --n 160 --preset fast` gate, plus the same
`evidence/n036_evidence.py` (24 checks) run against the actual shared tree.
Pre-flight run of the identical suite (login-node CPU + the isolated HEAD+mine
tree) already passed 24/24 - see Outcome below for the job exit codes.

## Outcome

**Both gates pass clean, all 8 flags landed.** GPU job 908569 (h11-16, 9m24s,
~0.16 GPU-h of the h100): `GATE_SUITE_EXIT=0` (4 configs' `test_model_parity`
+ geometry/stack/encoder/sampling parity + `test_design_quality --n 160
--preset fast`, 0/160 diverged) and `GPU_SUITE_EXIT=0` (plain regression
smoke test + the 24-check evidence suite, 24/24, + the standalone
`--mode rotation` reproduction). CPU job 908570 (h33-60, 6m38s):
`CPU_SUITE_EXIT=0`, same parity suite + the same 24/24 evidence suite (no
`--preset fast`/CUDA-only gate on CPU, matching `o8_ship_cpu.sbatch` /
N035's own precedent - symmetric generation is eager-only anyway, so nothing
about `--sym` needs CUDA specifically). Numbers reproduce byte-for-byte
between the pre-flight login-node run, the isolated HEAD+mine `git archive`
tree, and both cluster jobs.

Per-flag headline evidence (full numbers in both job logs and
`evidence/n036_evidence.py`'s own output):

| flag | portable? | evidence |
|---|---|---|
| `--mode rotation` | **NO - dead upstream** | `ScrewVP.replicate_pos()` TypeError reproduced with no jax dependency; `--mode rotation` raises `SystemExit` citing it, both via `parse_sym` directly and through the full CLI |
| `--fixcenter_threshold` | yes | `do_radius=True` places the subunit at the configured radius (centroid x=25.0 vs radius 25.0); `do_radius=False` does not (centroid x=-0.3); end-to-end a high threshold measurably changes the whole trajectory (33.34 A max coordinate diff vs baseline, GPU run) |
| `--compact_lr` | yes | unit: Rg 10.53->7.66 A; end-to-end: Rg 25.68->8.82 A (GPU) / 15.47->8.84 A (CPU) |
| `--clash_lr` | yes | unit: min cross-chain CA-CA 0.866->1.963 A; end-to-end: min cross-chain distance changes measurably both directions (GPU 5.47->4.35, CPU 4.08->4.37 - direction depends on the random baseline clash geometry, magnitude is the point) |
| `--mix_output` | yes (shares `--sym_mean`'s mechanism) | unit: couple vs select give different representative subunits from the same 4 noisy copies (1.957 A); end-to-end same-seed coordinate diff 6.63 A (GPU) / 50.90 A (CPU) |
| `--sym_noise` | yes | unit: forces exact cross-subunit symmetry when on (reconstructed-frame RMSD ~1e-8) vs unenforced when off; end-to-end same-seed coordinate diff 13.72 A (GPU) / 62.67 A (CPU) |
| `--f_small` / `--f_strand` | yes | a strict `--f_small 0.01` measurably triggers `rejected ... regenerating`; the default (1.0) never does, on both GPU and CPU |
| `config.symm` / `inference/symmetry.py::Screw` | **NO - dead, confirmed again** | neither `eval_sym.py` nor `eval_noise_schedule_benchmark.py` reads `config.symm`; `eval_sym.py` never imports `salad/inference/symmetry.py` at all (own textually-duplicated classes) |

## Outputs
- `/home/hpc/v181bb/v181bb10/SALAD_torch/salad_torch/inference/symmetry.py` (extended)
- `/home/hpc/v181bb/v181bb10/SALAD_torch/scripts/generate.py` (extended)
- `/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N036-symmetry-flags/upstream/eval_sym.py`
- `/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N036-symmetry-flags/upstream/inference_symmetry.py`
- `/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N036-symmetry-flags/evidence/n036_evidence.py`
- `/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N036-symmetry-flags/evidence/prove_screwvp_bug.py`
- `/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N036-symmetry-flags/reconstruction/build_mine.py`
- `/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N036-symmetry-flags/reconstruction/generate.py.mine.patch`
- `/home/hpc/v181bb/v181bb10/salad_jobs/n036_symmetry_gpu.sbatch` (job 908569)
- `/home/hpc/v181bb/v181bb10/salad_jobs/n036_symmetry_cpu.sbatch` (job 908570)

## Next
Nothing outstanding for this item. Two things worth a future agent's
attention, not blocking: (1) `--mix_output`/`--sym_mean` are now two CLI
names for one mechanism - could be unified/deprecated later if it ever
causes confusion; (2) the `_SYM_FILTER_MAX_RETRIES=20` safety cap is a
port-side deviation from upstream's unbounded retry, inert at the shipped
defaults but worth knowing about if `--f_small`/`--f_strand` ever get used
in anger on a hard target.
