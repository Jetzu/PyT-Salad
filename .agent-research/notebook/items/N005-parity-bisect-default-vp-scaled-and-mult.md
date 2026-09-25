# N005 · Parity bisect: default_vp_scaled and multimotif_vp fixtures were TF32

type: fix · date: 2026-09-25 · status: done

## Goal
Find the root cause of the N004 parity gap: `default_vp_scaled` 0.00589 A and
`multimotif_vp` 0.03694 A max coordinate deviation against JAX, versus a
0.00002 A floor for `default_vp` / `default_ve_scaled`, with `aatype` exact in
all four.

## Plan
Bisect inward from the block-0 output: dump every `DiffusionBlock`
intermediate on both backends on identical inputs, find the first tensor that
diverges. Control: run the same instrumentation on `default_vp`.

## Changes
- `scripts/gen_fixtures_model.py`: `jax.config.update("jax_default_matmul_precision",
  "highest")` before any computation; every fixture now records `jax_platform`
  and `jax_matmul_precision`.
- `tests/test_model_parity.py`: refuses a fixture whose recorded precision is
  not `highest`, or that predates the pin.
- `notes/04_PARITY_AND_PERF.md` section 5 rewritten (was "Open: two configs sit
  far above the parity floor").
- All four `fixtures/model_*.npz` regenerated on CPU with the pin.

## Evidence
All commands in `results/N005-parity-bisect/`. Container
`salad.sif`, params `/hnvme/workspace/v181bb10-salad/params/*.jax`, torch env
`miniforge3/envs/triad/bin/python`.

1. **The JAX reference was not reproducible.** `scripts/gen_fixtures_model.py`
   rerun on CPU reproduced `fixtures/model_default_vp.npz` **bit-identically**
   (0 diffs, all 37 arrays) but not `model_default_vp_scaled.npz`
   (`out_pos` 5.591e-03, `noised_pos_noised` 7.63e-06). Two CPU reruns of
   `default_vp_scaled` are bit-identical to each other, so CPU JAX is
   deterministic: the stored file came from elsewhere.
2. **Not the inputs.** Injecting the stored fixture's `noised_*` into a fresh
   CPU JAX run still left `out_pos` 5.577e-03 from the stored fixture
   (`inj_default_vp_scaled.npz`). `in_*` arrays were already bit-identical.
3. **The port is at the floor.** `jax_dump.py` / `torch_dump.py` dump 17
   tensors per block. torch vs the fresh CPU JAX run, same injected inputs:

   | | default_vp | default_vp_scaled |
   |---|---|---|
   | block0 `current_neighbours` / `cond_neighbours` | 0 / 3072 idx differ | 0 / 3072 idx differ |
   | block0 `pair_1` | 6.20e-06 | 2.38e-06 |
   | block0 `local_norm` | 4.77e-07 | 4.17e-07 |
   | block0 `block_out_pos` | 4.24e-05 | 2.10e-05 |
   | `final_pos` | 1.43e-05 | **9.27e-06** |

   No tensor diverges. The trunk is contractive, not chaotic: ~1e-5 in,
   ~1e-5 out.
4. **Provenance.** `results/N004-porting-gaps/gen_fixtures_task1.sbatch` shows
   the two failing fixtures were generated under
   `#SBATCH --partition=h100 --gres=gpu:h100:1`. The two passing ones were made
   on the CPU login node.
5. **Mechanism, measured (job 902051, h100, 3:30 elapsed,
   `gpu_902051.log`).** XLA:GPU lowers float32 `dot` to TF32 by default:

   | run | `out_pos` vs CPU | traj block0 | aatype |
   |---|---|---|---|
   | default_vp, H100, default precision | 5.98e-03 | 1.22e-02 | 0 differ |
   | default_vp, H100, `JAX_DEFAULT_MATMUL_PRECISION=highest` | 7.63e-06 | 2.38e-05 | 0 differ |
   | default_vp_scaled, H100, default | 5.95e-03 | 1.38e-02 | 0 differ |
   | default_vp_scaled, H100, `highest` | 3.41e-05 | 2.21e-04 | 0 differ |
   | default_ve_scaled, H100, default | 3.53e-03 | 1.86e-03 | 0 differ |

   `default_vp` - the config that always passed - reproduces the entire failure
   signature when regenerated on a GPU. Also: the old stored H100 fixture vs a
   fresh H100 rerun of the same config differs by 4.47e-03 A, so GPU fixtures
   are not self-reproducible either (autotuner kernel choice).
6. **Fix verified.** The pin is a bit-exact no-op on CPU (all four fixtures
   byte-for-byte unchanged, 0.000e+00). After regenerating on CPU:

   | config | before | after |
   |---|---|---|
   | default_vp | 0.00002 A | 0.00002 A |
   | default_ve_scaled | 0.00002 A | 0.00002 A |
   | default_vp_scaled | 0.00589 A | **0.00002 A** |
   | multimotif_vp | 0.03694 A | **0.00020 A** |

   All four `OK`. A GPU run made *after* the pin lands 4.7e-05 from CPU for
   `multimotif_vp` (`gpu_multimotif_vp.npz`, written 22:25:54 vs the generator
   patch at 22:24:59), so GPU generation is now safe.
7. **Guard verified.** Pointing the test at the old H100 file exits 1 with
   "predates the precision pin", not a 300x parity failure.

## Outcome
Not a port bug. `fixtures/model_default_vp_scaled.npz` and
`model_multimotif_vp.npz` were TF32 references produced on an H100 while the
port and the other two fixtures are fp32-on-CPU. Fixed by pinning
`jax_default_matmul_precision=highest` in the generator, recording backend
provenance in every fixture, and making the parity test refuse a fixture that
is not fp32-exact. All four configs now pass at the floor.

The N004 conclusion "the only difference is the weights ... points at the
weights themselves" is refuted: the weights were never involved.

## Outputs
`.agent-research/results/N005-parity-bisect/` (scripts + log; the 85 MB of `.npz` is at `/hnvme/workspace/v181bb10-salad/n005_evidence/` — see that folder's README.md)
- `jax_dump.py`, `torch_dump.py` — per-block intermediate dumps (17 tensors x 6 blocks)
- `jaxdump_*.npz`, `inj_*.npz`, `torchdump_*.npz` — the bisection data
- `regen_*.npz`, `new_*.npz` — CPU regenerations (pre/post pin)
- `gpu_*.npz`, `gpuhi_*.npz`, `gpu_check.sbatch`, `gpu_902051.log` — the H100 arm
- `ORIG_model_default_vp_scaled.npz`, `ORIG_model_multimotif_vp.npz` — the TF32 fixtures, kept as evidence

## Next
See TODO T7.
