# N035 · Multi-state design: --dssp parent/child_1/child_2 (upstream eval_split_change.py)

type: task · date: 2026-09-27 · status: done

## Goal
Port upstream salad's multi-state design task (`salad/training/eval_split_change.py`,
Lisanza *et al.* 2024) into this port's unified `scripts/generate.py`, extending the
existing single-state `--dssp` flag to accept `"parent/child_1/child_2"` rather than
adding a new script - per `notes/specs/06_generation_flags.md`'s existing convention and
this task's explicit instruction. Upstream has no copy in this repo; fetched from
`https://raw.githubusercontent.com/mjendrusch/salad/master/salad/training/eval_split_change.py`.

## Plan
Not an experiment (no baseline/control-arm comparison of a *pre-existing* behavior) - a
straight port. Read upstream in full first (done), identify the two independent mechanisms
(§1 secondary-structure conditioning per state, §2 the hard tertiary-structure coordinate
coupling `couple_part`/`SplitChange.update_pos`), then:
1. `options.py::parse_dssp_split` - string parsing (parent/child_1/child_2, same_mask,
   parent_split), extending the existing single-state `parse_dssp`.
2. NEW `inference/multistate.py` - the coupling geometry (`couple_part`,
   `update_split_change`), mirroring how `--sym`'s geometry lives in `symmetry.py` rather
   than `options.py`.
3. `scripts/generate.py::run_multistate` - a dedicated 3-state lockstep generation loop,
   reusing the EXISTING `sample_group` (built for `--group_size`, already exactly "N
   independent designs, one shared clock, per-design state") with `n=3` and a
   `step_fn_group` that applies the coupling before each state's noise+predict. No changes
   to `sampling.py`/`model.py`/`fast.py`.
4. Full spec write-up: `notes/specs/07_multistate_dssp.md` (upstream mechanism, port
   decisions, interactions).
5. Evidence: reproduce upstream's own Lisanza README example, generate parent+2 children,
   check (a) P-SEE secondary structure matches the requested string per state, (b) Kabsch
   CA RMSD at `same_mask` (tertiary-coupled) positions between each child and its parent
   segment, COUPLED run vs an UNCOUPLED control (three independent single-state runs on the
   same three DSSP strings) - the control is what isolates the tertiary mechanism from mere
   secondary-structure agreement (both arms share the same secondary-structure target).
6. Gate: full parity suite (4 configs + geometry/stack/encoder/sampling) + `test_design_quality
   --n 160 --preset fast`, GPU AND CPU, per this repo's PR gate.

## Changes
- `salad_torch/inference/options.py`: + `MultiStateDSSP` (dataclass), `parse_dssp_split`.
- `salad_torch/inference/multistate.py` (NEW): `couple_part`, `update_split_change`.
- `salad_torch/inference/__init__.py`: re-export the above (matches symmetry/template
  precedent).
- `scripts/generate.py`: `--dssp` help text updated; new `_MULTISTATE_INCOMPATIBLE` +
  `run_multistate`; `main()` dispatches to it before `apply_precedence_chain` when `"/"` is
  in `--dssp`.
- `notes/specs/07_multistate_dssp.md` (NEW): full spec, upstream mechanism transcribed,
  port decisions documented (length-match validation not in upstream; flag-incompatibility
  policy; per-step centering not reproduced, matching existing single-state precedent).
- `.agent-research/results/N035-multistate/evidence.py` (NEW): PSEA + Kabsch-RMSD analysis,
  self-tested Kabsch implementation, no torch/GPU dependency.

**Shared-tree note**: two sibling agents (shape-conditioning `--blobs`, a symmetry
optimiser) were editing `scripts/generate.py`/`salad_torch/inference/options.py`/
`salad_torch/inference/__init__.py` concurrently. Isolated my changes by reconstructing a
clean "HEAD + only my hunks" version of each shared file from `git show HEAD:<path>` +
the exact edits I made (verified against `git diff`'s hunk boundaries, which showed no
line-level overlap - only adjacent insertions), then staged that reconstruction with
`git apply --cached` rather than `git add <file>` (which would have swept up the
siblings' uncommitted work too). Verified: `git show :scripts/generate.py` /
`:salad_torch/inference/__init__.py` are byte-identical to the reconstruction, and both
compile. `salad_torch/inference/options.py` needed no such surgery - confirmed its entire
diff was mine alone (no sibling touched it as of this check).

## Evidence
GPU job 908519 (h100) + follow-up 908566 (fixed evidence script), CPU job 908520 (cpu
partition). Full numbers in `.agent-research/results/N035-multistate/RESULTS.md`.

Gate: **GATE_SUITE_EXIT=0** (GPU, job 908519) and **CPU_SUITE_EXIT=0** (job 908520) - parity
x4 configs + geometry/stack/encoder/sampling parity all EXIT 0 on both, `test_design_quality
--n 160 --preset fast` 0/160 diverged (GPU only, matches `o8_ship_cpu.sbatch` precedent of
skipping this GPU-only suite on CPU), ordinary single-state regression check EXIT 0 on both.

Feature evidence (Lisanza et al. 2024 construct, 6 designs, `default_vp-200k`, upstream's
documented recipe): PSEA secondary-structure agreement 0.837/0.892/0.913
(parent/child_1/child_2) vs the requested string. Tertiary-coupling Kabsch CA RMSD at
`same_mask` positions: child_1 COUPLED mean 0.356 A vs UNCOUPLED CONTROL mean 5.13 A
(~14x); child_2 COUPLED mean 0.411 A vs UNCOUPLED 5.41 A (~13x). The uncoupled control
(three independent single-state runs on the same three DSSP strings, no coupling) shares
the identical secondary-structure target, so the >10x RMSD gap isolates the tertiary
mechanism specifically - it is not an artifact of both arms drawing similarly-shaped
helices.

One bug found and fixed IN THE EVIDENCE JOB SCRIPT, not the product: job 908519's
uncoupled-control "parent" leg crashed (`ValueError: substring not found`) because the
Lisanza parent string contains lowercase segments and the single-state `--dssp` path is
case-sensitive uppercase-only by design (§5.4 of `06_generation_flags.md`) - `parse_dssp_split`
upper-folds each substring internally for the coupled run, but a bare single-state
`--dssp "$PARENT"` call does not do that itself. Fixed by upper-folding the control strings
in the job script (`n035_evidence_v2.sbatch`); reran as job 908566, `EVIDENCE_V2_EXIT=0`.

## Outcome
**DONE, gate-passing.** Both required suites pass on GPU and CPU with the feature landed;
the feature demonstrably does what upstream/Lisanza et al. describe (secondary structure
per state + a real, measured, order-of-magnitude tertiary-structure coupling effect, not
just a string parser). Nothing outside `inference/{options,multistate,__init__}.py` and
`scripts/generate.py`'s new `run_multistate` code path was touched, so the gate passing is
expected (a regression check, not a stress test of unrelated code) - the real test of THIS
feature is the evidence numbers above, which are the compelling part.

Could not reproduce/verify (see RESULTS.md "What I could not reproduce" for the full
reasoning): bit-exact JAX parity for this feature (not applicable - no cross-backend
fixture exists for `eval_split_change.py`, and this project's own standing position is that
no seed gives cross-backend bit-identical output regardless of feature); upstream's full
1000-design production scale (ran 6, sufficient for the evidence question asked); true
DSSP (used the port's own P-SEA CA-only tool as instructed, reused not reimplemented).

## Outputs
- `/home/hpc/v181bb/v181bb10/SALAD_torch/salad_torch/inference/multistate.py`
- `/home/hpc/v181bb/v181bb10/SALAD_torch/salad_torch/inference/options.py` (extended)
- `/home/hpc/v181bb/v181bb10/SALAD_torch/salad_torch/inference/__init__.py` (extended)
- `/home/hpc/v181bb/v181bb10/SALAD_torch/scripts/generate.py` (extended)
- `/home/hpc/v181bb/v181bb10/SALAD_torch/notes/specs/07_multistate_dssp.md`
- `/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N035-multistate/evidence.py`
- `/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N035-multistate/RESULTS.md`
- `/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N035-multistate/n035_evidence_v2_908566.json`
- `/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N035-multistate/logs/` (all 3 job logs)
- `/home/hpc/v181bb/v181bb10/salad_jobs/n035_multistate_gpu.sbatch` (job 908519)
- `/home/hpc/v181bb/v181bb10/salad_jobs/n035_multistate_cpu.sbatch` (job 908520)
- `/home/hpc/v181bb/v181bb10/salad_jobs/n035_evidence_v2.sbatch` (job 908566)

## Next
- `LINT DUE` was flagged (19 items since last lint) when this item was created via
  `new-item.sh`; **deliberately not run** by this agent - two sibling agents have live
  uncommitted edits in the shared notebook (`INDEX.md` in particular) and a lint pass
  consolidates/rewrites across the whole notebook, which risks clobbering their in-flight
  work. Logged as a conservative deferral (see `SETUP_NOTES.md`), not an oversight; whoever
  lands next after all three sibling tasks have committed should run it.
- No further work planned for this feature; upstream's `--variants` (restart-branching at
  an out-step) was deliberately not ported for EITHER the single- or multi-state path (out
  of this port's existing scope, not something this task asked for).

## Post-commit note
Landed as 3 commits: `fd7737a` (feature), `7e23c99` (fix: a plain `git add` re-staging step
after a concurrent index write reset `salad_torch/inference/__init__.py` picked up the
blobs sibling's still-uncommitted `from .blobs import ...` line too - caught immediately by
a `git archive`-based isolated-checkout import check, fixed via `git hash-object`/
`update-index` rather than another plain `git add`), `1856da7` (INDEX row commit hash).
Final isolated-checkout verification (`git archive HEAD | tar -x` into a scratch dir,
fresh `python -c "import scripts.generate"`) confirms both this feature and sibling N036's
work import and compile cleanly from a clean checkout of the final HEAD.
