# N003 · Red-team pass: fast path, symmetry, encoder, generation flags

type: research · date: 2026-09-25 · status: done

## Goal
Adversarial pass (red team) over the newest, least-tested code: the compiled
`modules/fast.py` inference path (including the UNCOMMITTED condition cache +
static input buffers), `inference/{symmetry,template,postprocess,options}.py` +
`scripts/generate.py`, `modules/encoder.py`, and `data/` + `training/`.
Find crashes, and above all silent wrongness / divergence from JAX.
Source read-only; the main agent lands the fixes.

Note: the red team should be Codex (different model family, ORCHESTRATION.md),
but Codex produces no output right now, so this pass was **same-family
(Claude)**. Treat the priors as correlated with the implementer's.

## Plan
Re-run the three claimed baselines first, then attack each target with an
explicit expected-vs-observed and a minimal reproducer. JAX reference run in
`salad.sif`; torch in the `triad` env; GPU work via slurm h100.

## Changes
None — no source file was edited. Two test-local shims (a restored
`Rigid.__getitem__`) live inside the reproducer scripts only.

## Evidence
Full write-up with every number:
`.agent-research/results/N003-red-team-pass-fast-path-symmetry-encoder/FINDINGS.md`

Baselines: geometry 38/38 PASS; model parity 0.00002 A on `default_vp` and
`default_ve_scaled` PASS; `tests/test_stack_parity.py` **crashes** (D2).

GPU: job **901501** (h100, 00:24 elapsed, ~0.4 GPU-h), log `r03_901501.log`.
901487 was a first attempt killed by a bug in my own harness; 901569 was a
redundant split job, cancelled. Total ~0.9 GPU-h.

Reproducers r01–r08 in the same folder; `## How to reproduce` at the end of
FINDINGS.md has the exact commands.

## Outcome
2 hard crashes + 11 silent-wrongness / divergence findings; 8 attacks found
nothing. Ranked headlines:

- **D1 (crash, total).** `modules/encoder.py:302,372` still use
  `frames[:, None]`; `Rigid.__getitem__` was removed in `dda09aa` ("all call
  sites updated" — these two were missed). Every Encoder call raises
  `TypeError`. Invisible because the Encoder had no test and
  `load_encoder(strict=True)` only checks the state dict. Now load-bearing:
  the concurrent N004 edit to `training/loop.py:205` calls the Encoder from
  `prepare_batch`. Fix: `frames.unsqueeze_batch(1)`.
- **S1 (worst silent).** `FastDenoiser._cond_cache` is keyed on nothing.
  Conditioning added after the first call is ignored: dssp 5.52 A wrong,
  dmap/template 15.34 A wrong, dssp_mean 4.93 A wrong, chain_index 2.53 A
  wrong — each matching the STALE answer to ~5e-4 A. Two designs in one
  process with different `dssp_condition` differ by 1.7e-4 A when the correct
  answer is 11.4 A. Not user-facing today (generate.py never builds a
  FastDenoiser) but armed.
- **S2.** `FastDenoiser` adopts the caller's tensor as a static buffer
  whenever the blank slot is `None` or the shape differs, then `copy_`s into
  it next call — measured: caller A's `cyclic_mask` flipped 0/64 -> 64/64.
- **S3.** `clash_repulsion_grad` clamps `dx**2` where JAX clamps `dx`;
  100% relative error, corr 0.95, and 5.97e-3 A per call inside the
  `--sym` feedback loop for every step with `raw_t > 0.5`.
- **S4.** `--cloud_std` is a silent no-op on `default_vp`: the port's
  `apply_diffusion` gives `vpfixed` priority over `data["cloud_std"]`;
  upstream lets data win. Byte-identical PDB for unset / 0.0 / 99.0.
- **S5/S6.** `FastDenoiser` ignores `predict_aa=False` when building
  `atom_pos` (7.73 A inconsistent with its own returned `aatype`) and ignores
  `aa_condition` (12/64 residues).
- **S7/S8.** `--dssp random` / `--dssp_mean random` use the unseeded stdlib
  `random` (15–25 A between identically-seeded runs); the Gumbel neighbour
  draw never receives `--seed` either (0.004–0.019 A at 60 steps).
- **S9/S10.** `--sym` with `count ∤ L` orphans the tail (11.66 A CA–CA break
  for `--num_aa 100 --sym c12`); `c1` is a no-op in the port but not in JAX.
- **S11.** Compiled vs eager 5.9e-4 A at matmul="highest", and the compiled
  path is not even self-reproducible (2.5e-4 A between identical calls).

Found nothing: `inference/template.py` (motif recovery 0.099 A CA RMSD),
`screw_replicate` wherever `count | L` and `chain_center=False` (1.5e-5 A vs
JAX), C4 symmetry on the written PDB (0.024–0.031 A, axis ±Y, angle 89.99/
179.99/90.02 deg — better than the claimed 0.04–0.07 A), held FastDenoiser
outputs, `data/collate.py`.

Bonus: built the Encoder parity test that did not exist. With the one-line D1
fix the Encoder is **exact** vs upstream `NSB.Encoder` on real weights
(local rel <= 1.9e-6, coords max 0.00001 A, backbone[0:5] bit-exact, 3 shapes)
and robust on 13 degenerate inputs. `r05_encoder_parity_{jax,torch}.py` should
be promoted to `tests/test_encoder_parity.py`.

Caveat: `training/loop.py` was being edited by N004 during this pass, so
target 4 (data + training) is only partly covered — no batch-of-two vs
separate-loss test was run.

## Outputs
`.agent-research/results/N003-red-team-pass-fast-path-symmetry-encoder/`
- `FINDINGS.md` — the full report
- `r01_clash_grad_{torch,jax}.py` — S3
- `r02_screw_{torch,jax}.py` — S9, S10, S3 end-to-end
- `r03_fast_attack.py`, `r03.sbatch`, `r03_901501.log` — S1, S2, S5, S6, S11, N1–N3
- `r04_encoder_stress.py` — D1 + degenerate-input sweep
- `r05_encoder_parity_{jax,torch}.py`, `r05_encoder_ref.npz` — new parity test
- `r06_sym_residual.py`, `r06b_sym_axis.py` — C4 residual (no defect)
- `r07_remainder_drift.py` — S9
- `r08_*` — redundant split job, cancelled

## Next
See notebook/TODO.md T1–T6.
