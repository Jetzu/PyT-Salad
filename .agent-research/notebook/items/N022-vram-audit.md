# N022 · VRAM requirement table for every inference mode up to L=1000

type: task · date: 2026-09-26 · status: done

## Goal
Measure peak VRAM (torch.cuda.max_memory_allocated AND max_memory_reserved)
for every implemented inference mode (eager, --fast x {exact,fast,deterministic}
presets, --group_size sweep, --sym, --renoise, --sample_final) at L up to 1000,
plus the training path's own footprint for reference. Fit linear vs quadratic
models per mode, give practical per-card limits, flag pathological modes.
Measurement only - no model/script code changes (two other agents, N020/N021,
own basic.py/fast.py inductor config and generate.py/sampler/grouping code
concurrently).

Pinned snapshot: `git archive fd9ba32 -- salad_torch scripts` extracted to
/hnvme/workspace/v181bb10-salad/n022tree (80 files, commit fd9ba325199b).
Builds on N019's .agent-research/results/N019-multidesign-occupancy/RESULTS.md
section 4.4 (group-size marginal VRAM, L<=400) and `memory_bound_group` in
salad_torch/modules/fast.py.

## Plan
63 cells = one (mode, L, N) per process, via a new harness
(`_probe/harness.py`, lived only on the /hnvme snapshot, not committed) that
imports the snapshot's real `scripts/generate.py::main()` /
`salad_torch.training.train::main()` in-process (so `torch.cuda`'s peak
stats see the run) with `--num_steps 3 --out_steps 2` (smallest schedule that
still reaches one out-step). 8 SLURM jobs (904365-904372, h100, 1 GPU each),
grid: eager/fast-exact/fast-fast/fast-deterministic x L in
{50,100,200,300,400,600,800,1000} (32 cells), group_size in {1,2,4,8,16(,24,
32 at L=100)} x L in {100,400,1000} (17+4 cells), sym/renoise/sample_final x
3 L's each (9 cells), training reference (1 cell). Estimated ~7 GPU-h before
running (smoke test showed --fast compiles cost 250-370s/process); actual
~3.9 GPU-h.

## Changes
None. No model/script file touched (T4/red-team scope respected: N020 owns
basic.py + fast.py inductor config, N021 owns generate.py/sampler/grouping).

## Evidence
Jobs 904327 (smoke test, 4 cells, validated harness end to end) and
904365-904372 (full sweep, 63 cells, all COMPLETED exit 0:0, zero OOMs/errors
on this 93.1 GiB H100). Raw JSON (peak_allocated_bytes, peak_reserved_bytes,
wall_s, compile_capture_s) for every cell: `raw/*.json`. Full analysis,
fitted models, per-card table: `RESULTS.md`.

## Outcome
Linear (not quadratic) in L holds for every mode through L=1000 — the a
priori quadratic-blowup risk (pair-shaped distogram/DETSEG indicator) does
not appear in peak memory. DETSEG hypothesis in the task brief REFUTED:
`deterministic` preset is bit-identical to `exact` in peak reserved at every
L, single-design and grouped (N019's per-L=50..400 marginal ~3.8 MiB/aa/design
now CONFIRMED flat through L=1000: 4.19/3.89/4.00 MiB/aa/design at
L=100/400/1000). Headline finding instead: reserved/allocated ratio grows
from 2.1x (N=1) to **14.1x** (group_size=16, L=1000) — reserved, not
allocated, is the number that sizes a card. `--fast` costs MORE memory than
eager (two captured graph variants resident at once). Found and reported (not
fixed, out of scope): the N017/O1 CSE post-pass installs an unpicklable
closure that silently breaks inductor's on-disk FX-graph cache, so every
`--fast` process pays a full 4-8 minute recompile with zero cross-process
reuse via that cache (a separate, lower-level Triton kernel cache does still
help on repeat-L within one job).

## Outputs
- `.agent-research/results/N022-vram-audit/RESULTS.md` — full table, fits,
  per-card table, pathology list.
- `.agent-research/results/N022-vram-audit/raw/*.json` — 63 per-cell
  measurements.
- Snapshot `/hnvme/workspace/v181bb10-salad/n022tree` (commit fd9ba325199b)
  deleted after use, per the inode-grace rule.

## Next
- The inductor FX-graph-cache pickling bug (`_cse_pass` closure,
  `salad_torch/modules/fast.py`) is real and unfixed — belongs to N020/O1's
  files, out of this task's scope.
- Group axis was measured at 3 L's (100/400/1000), not the full 8-length
  grid — if a future task needs the group curve at L=150/200/300/600/800 too,
  the combined model in RESULTS.md §2.1 (reserved(L,N) ≈ (5.73L+235) +
  (N-1)*4.00L MiB) already interpolates those within ~1-4% based on what was
  measured.
