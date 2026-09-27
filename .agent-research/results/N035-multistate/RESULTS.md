# N035 — Multi-state design (`--dssp "parent/child_1/child_2"`) — results

## Gate suites (PR gate)

| Suite | GPU (job 908519, h100) | CPU (job 908520, cpu) |
|---|---|---|
| `test_model_parity.py` x4 configs (default_vp, default_vp_scaled, default_ve_scaled, multimotif_vp) | EXIT 0 each | EXIT 0 each |
| `test_geometry_parity.py` | EXIT 0 | EXIT 0 |
| `test_stack_parity.py` | EXIT 0 | EXIT 0 |
| `test_encoder_parity.py` | EXIT 0 | EXIT 0 |
| `test_sampling_parity.py` | EXIT 0 | EXIT 0 |
| `test_design_quality.py --n 160 --preset fast` | EXIT 0 (0/160 diverged) | not run (needs CUDA, matches `o8_ship_cpu.sbatch`'s own precedent) |
| ordinary single-state `scripts/generate.py` (regression check) | EXIT 0 | EXIT 0 |
| **`GATE_SUITE_EXIT`** | **0** | **0** (see `CPU_SUITE_EXIT=0`) |

Full logs: `logs/n035_ms_gpu_908519.log`, `logs/n035_ms_cpu_908520.log`. Nothing this
feature added touches `sampling.py`/`model.py`/`fast.py`/`modules/*` - these suites verify
that stays true.

CPU job additionally ran a small multi-state smoke test (parent 14aa / child_1 8aa /
child_2 6aa, 20 steps) purely to confirm the feature is device-agnostic (no CUDA-only
assumption anywhere in `run_multistate`/`multistate.py`) - it produced all three PDBs,
`EXIT 0`.

## Evidence: does the feature do what it claims?

Two jobs: 908519's evidence leg crashed on the uncoupled-control "parent" call with
`ValueError: substring not found` - a bug in the **evidence job script**, not the port:
the single-state `--dssp` path is case-sensitive uppercase-only by design
(`notes/specs/06_generation_flags.md` §5.4 - reproducing upstream's own `str.index`
failure mode), and the Lisanza parent string contains lowercase `l`/`e` segments that
`parse_dssp_split` upper-folds internally for the coupled run but that a bare single-state
`--dssp "$PARENT"` call does not. Fixed in `n035_evidence_v2.sbatch` (upper-fold the
uncoupled-control strings with bash `${VAR^^}`) and reran (job 908566, `EVIDENCE_V2_EXIT=0`).
Numbers below are from 908566 (`n035_evidence_v2_908566.json`); PSEA numbers from 908519 and
908566 agree to within design-to-design noise (both are real generations, just different
random seeds for the failed-then-fixed control leg), confirming the crash was isolated to
the one uncoupled-parent call and didn't taint the coupled run's own numbers.

Setup: upstream's own README (Lisanza *et al.* 2024) example construct, `default_vp-200k`
checkpoint, 500 steps / out_step 400 / `prev_threshold 0.8` / `cosine(t)` - exactly
upstream's documented recipe, `--num_designs 6` (not upstream's 1000 - a feature-validation
run, not a benchmark). parent=101aa, child_1=53aa, child_2=48aa, 52 tertiary-coupled
(`same_mask`) residues total (26 in each child).

### 1. Secondary structure (P-SEA CA-only assignment vs the requested string, `_` skipped)

| state | mean agreement |
|---|---|
| parent | 0.837 |
| child_1 | 0.892 |
| child_2 | 0.913 |

All three states independently track their OWN requested secondary structure at
82-91% agreement per design (P-SEA is itself an approximation of true DSSP, so 100% was
never the right target) - the model is honoring each state's distinct H/E/loop pattern, not
just producing a generic fold.

### 2. Tertiary coupling — Kabsch CA RMSD at `same_mask` positions (Angstrom)

| | COUPLED (n=6) | UNCOUPLED CONTROL (n=6) |
|---|---|---|
| child_1 vs parent[:53] | mean 0.356, median 0.346 | mean 5.13, median 5.30 |
| child_2 vs parent[53:] | mean 0.411, median 0.393 | mean 5.41, median 4.30 |

The **uncoupled control** is three fully independent single-state generations off the same
three DSSP strings (so the SAME secondary-structure target as the coupled run, just no
`update_split_change` coupling at any step) - this isolates the tertiary mechanism from mere
secondary-structure agreement, which alone would not explain sub-angstrom RMSD (two
independently-folded helix-loop-helix motifs sharing a secondary-structure label are not
generically superimposable to <0.5 A; the control arm measures exactly that null and finds
~5 A, an order of magnitude larger).

**Reading the ~0.35-0.41 A residual (not exactly 0):** `update_split_change` overwrites a
child's `same_mask` coordinates with the parent's every step EXCEPT it is never re-applied
after the final out-step's own noise+predict call (the coupling runs on the *incoming*
`pos` at the top of each step, so the reported/written snapshot is one free denoising step
past the last coupling application - see `notes/specs/07_multistate_dssp.md` §1,§4). A
literal 0.0 A would indicate the coupling was applied post-hoc to the written output (a
much weaker, cosmetic check); this small nonzero residual is the expected signature of a
constraint enforced *during* denoising rather than after it.

## What I could not reproduce / verify

* Bit-exact parity with the JAX reference for this feature specifically: not applicable -
  this whole port has no JAX-vs-torch fixture for `eval_split_change.py` (upstream is JAX,
  the port is torch, and per this project's own standing note, no seed gives cross-backend
  bit-identical output regardless of feature). The parity SUITES that do exist (§ above) are
  all untouched by this change and still pass, which is the relevant regression check.
* Upstream's own 1000-design production run (README example uses `--num_designs 1000`) -
  ran 6 for evidence, not 1000; a full-scale run was not requested and would cost
  proportionally more GPU time for no additional evidence value at this stage.
* A REAL DSSP (not P-SEA) comparison - this port's own CA-only P-SEA tool
  (`N011-fast-preset/ss.py`) was reused as instructed; true DSSP needs hydrogens/a full
  atomic model this port's backbone-focused output doesn't reliably provide.
