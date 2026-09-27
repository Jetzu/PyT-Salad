# N034 · Shape conditioning: --blobs/--blob_scale (upstream eval_ve_shape.py)

type: task · date: 2026-09-27 · status: done

## Goal

Port upstream's `salad/training/eval_ve_shape.py` (`--blobs`/`--blob_scale`,
plus the `--replace_threshold` mechanism it needs) into `scripts/generate.py`,
the single entry point. VE-model only. Third of three parallel gap-closing
ports (siblings: multi-state design, symmetry optimiser flags) into the same
file - keep the diff minimal/localised, per the brief.

## Plan

Fetched upstream source directly (no local copy existed):
`curl -sfL https://raw.githubusercontent.com/mjendrusch/salad/master/salad/training/eval_ve_shape.py`
+ its 4 example CSVs under `letters/`. Read it in full before writing
anything. Two independent mechanisms to reproduce:

1. `parse_blob`: CSV -> initial point cloud (`data["pos"]`), flat
   `resi`/`chain` (no `parse_num_aa` reuse, no cyclic, one unnumbered chain).
2. The inline per-step "hard replace" in `__main__` (`pos_centers`,
   hardcoded `reshape(-1, 100, 3)`), gated by `--replace_threshold`.

Full analysis in `notes/specs/08_shape_conditioning.md`.

## Changes

* `salad_torch/inference/blobs.py` (new): `parse_blobs`, `replace_to_blob_anchor`.
* `salad_torch/inference/__init__.py`: export both.
* `scripts/generate.py`: `--blobs`/`--blob_scale`/`--replace_threshold` CLI
  flags; `apply_blob_chain` (mirrors `apply_precedence_chain`'s contract,
  raises `SystemExit` for `--config` not `default_ve_scaled`, and for
  `--dssp`(literal/random)/`--template`/`--order_agnostic`/`--sym`/
  `--group_size`/`--batch_size`); one `data["pos"] = blob_init_pos.to(device)`
  override per design; `post_step_fn` extended to also call
  `replace_to_blob_anchor` (mutually exclusive with `--sym` by construction,
  so evaluation order between the two branches never matters).
* `notes/specs/08_shape_conditioning.md` (new): full spec, upstream quirks
  reproduced (header-skip, hardcoded chunk=100, the `--dssp` ambiguity and
  why it's rejected rather than guessed at).
* Coordinated with concurrent siblings via `git status`/`git diff` before
  every edit to `scripts/generate.py` (multistate agent's `run_multistate`/
  `_MULTISTATE_INCOMPATIBLE`/dispatch already landed in the shared tree by
  the time this item started) - diff is additive only, verified no lines of
  theirs touched.

## Evidence

Unit/CPU smoke (interactive, triad env, before submitting SLURM):
* `parse_blobs` on `s_shape.csv` * 1.8: `block0 CA == 1.8*(0,-11.217,17.200)`
  exactly; `replace_to_blob_anchor` raises `RuntimeError` on a non-100-
  divisible input (reproduces upstream's unguarded `reshape` crash).
* All 5 incompatibility guards (`--config default_vp`, `--dssp LHHH`,
  `--sym c4`, `--group_size 2`, `--template ...`) raise the intended
  `SystemExit` with a clear message.
* `--blobs` + `--dssp_mean`/`--merge_chains`/`--replace_threshold` compose
  without crashing.
* A short real run (`l_shape.csv`, 2 designs, 10 steps, CPU): after only 10
  steps (replace active every step, default `--replace_threshold 0.0`),
  per-block centroid error 0.022-0.044 A, placement accuracy 1.000 vs a
  deranged-label control's 0.000 (chance 0.200) - confirms the metric and
  the mechanism both work; NOT yet evidence of what the network does
  UNAIDED by replace over a full 500-step trajectory (that's arm B below).

SLURM jobs (submitted 2026-09-27, branch `n006-manual-cudagraph`,
`/home/hpc/v181bb/v181bb10/SALAD_torch`, GPU-h estimate stated before
submit: 1x H100 x <=3h requested <=3 GPU-h, far under the 25 GPU-h gate):

* GPU 908512 (h100, `salad_jobs/n034_gate_gpu.sbatch`): full parity suite (4
  configs) + `geometry/stack/encoder/sampling_parity` + `test_design_quality
  --n 160 --preset fast` + `--n 64 --preset exact` + the 5 guard checks +
  `--blobs`+`--fast` smoke + 3 real evidence arms (A: `l_shape`, replace on,
  8 designs/500 steps; B: `l_shape`, replace off via
  `--replace_threshold 1.1`, same seed/8 designs, the ablation; C: `a_shape`,
  8 blobs/L=800, replace on, 4 designs, generalisation check).
* CPU 908513 (48 cores, `salad_jobs/n034_gate_cpu.sbatch`): parity suite +
  eager de-novo smoke + the 5 guard checks + compose smoke + one CPU-feasible
  short evidence run (`s_shape.csv`, 2 designs, 60 steps).

NB: this task's cited "working example" `salad_jobs/o8_ship_cpu.sbatch` does
NOT parse (`bash -n` -> "unexpected end of file", missing 2 `done`s) -
flagged, not fixed (not this item's file); both `n034_gate_*.sbatch` are
independently written and `bash -n`-clean.

Backgrounded `~/.claude/skills/work-log/bin/wait-job.sh 908512 908513`.

## Outcome

**908512/908513 (first GPU/CPU run) found a real bug pre-commit.** GPU job
908512: parity suite + design quality PASSED (EXIT 0 throughout), but every
`--blobs` path that actually reaches `replace_to_blob_anchor` (i.e. the
default `--replace_threshold 0.0`) crashed:
`RuntimeError: Expected all tensors to be on the same device, but found at
least two devices, cuda:0 and cpu!`. Root cause: `blob_init_pos` (from
`apply_blob_chain`) was moved to `device` only inside the per-design
`data["pos"] = blob_init_pos.to(device)` line - the OUTER variable stayed on
CPU, and `post_step_fn`'s closure captured that same (still-CPU) outer
variable. `--replace_threshold 1.1` (arm B, disables the correction since
`raw_t` never reaches 1.1) never called the buggy function and so ran fine,
which is exactly why plain interactive CPU testing beforehand (device
always "cpu", so `.to(device)` is a no-op either way) never caught it - a
CPU-only smoke test cannot distinguish "moved to the right device" from
"never needed to move". Fixed with a one-line relocation: `blob_init_pos =
pc["blob_init_pos"].to(device)` right after `apply_blob_chain` returns
(`scripts/generate.py`), instead of leaving the move to each design's
`data["pos"]` copy alone.

Re-verified end to end at commit time by reconstructing the tree state (see
Changes): re-ran `bash -n` + `python -c "import ast; ast.parse(...)"` on the
reconstructed file, then resubmitted both gate jobs against the fixed live
tree: **GPU 908578 (`GPU_SUITE_EXIT=0`, 27m30s) and CPU 908579
(`CPU_SUITE_EXIT=0`, 7m47s)**, both clean.

Parity + design quality (unaffected, confirming no regression):
`model_parity` x4 configs, `geometry/stack/encoder/sampling_parity` all
EXIT 0; `test_design_quality --n 160 --preset fast` 0/160 diverged
(CA-CA 3.7623, Rg 13.01) and `--n 64 --preset exact` 0/64 diverged
(CA-CA 3.7630, Rg 12.65) - matching the recorded c5ab7a0/N015 baseline
(3.76/12.9) to within noise.

Shape-conditioning evidence (GPU h100 908578, 500 steps/design,
`--replace_threshold` default unless noted):

| arm | shape | blobs x aa | n | replace | centroid err mean (max), A | placement acc | vs deranged control | chance |
|---|---|---|---|---|---|---|---|---|
| A | l_shape | 5x100=500 | 8 | ON (0.0) | 0.016 (0.036) | 0.968 | 0.006 | 0.200 |
| B | l_shape | 5x100=500 | 8 | OFF (1.1) | 7.740 (13.21) | 0.799 | 0.029 | 0.200 |
| C | a_shape | 8x100=800 | 4 | ON (0.0) | 0.013 (0.025) | 0.958 | 0.009 | 0.125 |
| CPU | s_shape | 8x100=800 | 2 (60 steps only) | ON (0.0) | 0.003 (0.005) | 0.991 | 0.002 | 0.125 |

Reading it: placement accuracy (nearest-of-K-centres classification) is
~0.80-0.97 in EVERY arm, 4-8x the chance rate, and the deranged-label
control collapses to ~chance-or-below (0.006-0.029) in every arm - so the
generated residues really do sit nearer their OWN assigned blob than any
other, not an artefact of the metric. What the hard replace (`--replace_
threshold`) actually buys is **absolute** positional fidelity: with it OFF
(arm B) the design still knows roughly which blob it belongs to (0.799 vs
0.029) but drifts ~7.7 A from the literal requested coordinate; with it ON
(arms A/C/CPU) that error is <0.04 A, i.e. the network's own local geometry
is being translated to sit almost exactly at the CSV coordinate every
qualifying step. Rg confirms the qualitative picture too: 24.9-32.9 A for
these elongated multi-blob layouts vs 12.9-13.0 A for plain de-novo 100-mers
in the SAME parity run - these are not generic globular proteins.

Not reproduced/out of scope, and why (see `notes/specs/08_shape_conditioning.md`
§3 for the full reasoning): `--sym`/`--template`/`--order_agnostic`/
`--group_size`/`--batch_size` combined with `--blobs` (rejected,
`SystemExit`, upstream never exercises or doesn't declare these
combinations); `--dssp` literal/`"random"` combined with `--blobs`
(rejected - upstream's own dssp branch for this script is narrower than the
general precedence chain's, and either reading silently orphans the blob
point cloud's shape against a resized `num_aa`).

## Outputs

`.agent-research/results/N034-shape-conditioning/`: `a_shape.csv`,
`d_shape.csv`, `l_shape.csv`, `s_shape.csv` (upstream's example CSVs),
`x_evidence.py`, job logs (`j_*_908512/908513.log` = the buggy run,
`j_*_908578/908579.log` = the fixed/final run), `evidence_*.json` summaries,
one representative PDB per evidence arm. `salad_jobs/n034_gate_gpu.sbatch`,
`salad_jobs/n034_gate_cpu.sbatch` (independently written; the brief's cited
"working example" `salad_jobs/o8_ship_cpu.sbatch` does not parse - missing
2 `done`s, `bash -n` fails - flagged, not fixed, not this item's file).

## Next

None outstanding for this item. Possible follow-ups (not done here, out of
scope): wiring `--dssp_mean`-style broadcast conditioning tests at n>160 for
statistical power; deciding whether a future script should let `--blobs`
compose with `--group_size`/`--batch_size` (would need `post_step_fn`
support in `sample_group`, a `sampling.py` change this task's scope
(`salad_torch/inference/` only) excludes).
