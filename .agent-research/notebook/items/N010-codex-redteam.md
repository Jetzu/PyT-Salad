# N010 · Red-team: PyTorch SALAD port, fast path

type: research · date: 2026-09-26 · status: backfilled (done)

Backfilled by work-lint 2026-09-26: `results/N010-codex-redteam/` existed
with a full `FINDINGS.md` but no INDEX row or item file.

## Goal
Independent red-team pass over the committed `n006-manual-cudagraph` fast
path and the verification suite itself: find silent failures, not just
crashes, and check whether the existing tests would actually catch them.

## Plan
Audit each public entry point for real vs vacuous test coverage, then attack
the fast-path cache/capture logic, the symmetry hooks, the loss-checker
harnesses, and fixture provenance directly with targeted reproducers. No
source files edited; findings-only, scoped to this results folder.

## Evidence
Base commit `31937680e2a6b477d2d4bc94dee5959d79b2f0ec`. Concurrent edits to
`fast.py`/`diffusion.py`/`generate.py` appeared in the live worktree during
the audit; findings below are grounded in the committed version. `pytest
tests` collects 0 tests (`pytest_exit=5`) despite 5 `test_*.py` files passing
when run directly as scripts.

Ranked findings (full detail + reproducers in `FINDINGS.md`):
- **S1 (critical)**: `FastDenoiser`'s conditioning cache key is a plain sum
  over each conditioning tensor. Swapping DSSP labels that sum-collide
  returns the PREVIOUS design's denoised coordinates: fast-vs-eager diverged
  2.562 A on H100 with a pinned identical `uniform`, vs 7.6e-6 A when the
  cache key doesn't collide and 1.1e-5 A after explicit invalidation.
- **S2 (high)**: a captured `FastDenoiser` graph silently accepts a new
  sequence length and returns the OLD length's output (length-48 capture fed
  length-49 input returned 48 residues).
- **S3 (high)**: the symmetry hook constrains `snapshot["pos"]` but the PDB
  writer reads the unconstrained `snapshot["atom_pos"]`, so the written PDB
  is not actually symmetric (toy case: 4.77e-7 A on the constrained field vs
  2.208 A on the written atoms). Reproduced as an upstream-inherited bug, not
  a new port divergence.
- **S4 (high, verification gap)**: `tests/harness/check_loss{,2}.py` print
  `FAIL` for a corrupted reference but return exit 0 - no assertion actually
  gates the 12 loss terms.
- **S5 (medium, verification gap)**: model-fixture provenance (backend/
  precision fields) is self-asserted and only present on the 4 `model_*.npz`
  fixtures; a CPU-generated fixture relabelled as `gpu`/`highest` is accepted
  by `test_model_parity.py`, as is a fixture with `out_raw_angles` perturbed
  by +0.01 (inside its declared 1e-4 relative tolerance).
- **S6 (medium)**: symmetry leaves unconstrained tails (order > L, or last
  residue at L=9/C4) unchanged, and C1 skips upstream's centering step.
- **S7 (low)**: symmetry clash cost squares-then-clamps instead of
  clamps-then-squares (same defect class as K005 from N003, different site -
  the `chain_center` hook rather than the pairwise clash cost); gradient
  differs by up to 6.76e-5.
- **S8 (low)**: `FastDenoiser(compute_distogram=True)` is accepted but never
  produces a distogram key.
- **C1/C2 (low, crashes)**: `--sym c0` divides by zero; the synthetic
  training data source raises on the documented minimum `--num-aa 16`
  (`numpy.random.randint(16, 16)` is an empty interval).
- Several attacks found NO failure (documented, not just omitted): repeated/
  interleaved pinned fast-path calls, batch-duplication and masked/length-1
  loss edge cases, and 4 small PDB-writer layouts all matched upstream
  byte-for-byte or within noise.

## Outcome
Two high-severity silent-wrongness bugs in the fast path (S1 cache
collision, S2 stale-length capture) plus a real symmetry/PDB-writer output
bug (S3) and four verification-suite gaps (pytest collects nothing; loss
checkers can't fail; fixture provenance is forgeable; several public entry
points, incl. `sample`, training/data/checkpoint plumbing, and the PDB
writer, have no direct test at all). No fix applied here (findings-only,
per this task's write boundary) - a fix/verification plan is given in
`FINDINGS.md` but is not yet reflected in TODO.md or KNOWLEDGE.md.

## Outputs
`results/N010-codex-redteam/`: `FINDINGS.md` (full findings + fix/
verification plan), `inventory.log`, per-finding `repro_*.py`/`repro_*.log`/
`repro_*.sbatch`, `test_*.log` (direct test-script runs), `pytest.log`,
`generated/`, `generated_sym_template/` (CLI smoke PDBs), and 4 corrupted/
forged `*.npz` fixtures used as S1/S4/S5 attack inputs (each <2.2 MB, under
the notebook's 5 MB rule).

## Next
Findings not yet triaged into TODO.md/KNOWLEDGE.md by a Claude integrator
(per the four-role workflow, Codex red-team output needs a landing pass).
S1 and S2 in particular are live-code bugs in a path that may already be
committed downstream (N009/N011 build on the fast path) - worth reproducing
first-hand before any fix, per the reviewers-report/you-decide rule.
