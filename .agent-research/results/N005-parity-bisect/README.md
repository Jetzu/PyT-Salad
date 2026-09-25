# N005 evidence

Large arrays (85 MB of `.npz`) live at
`/hnvme/workspace/v181bb10-salad/n005_evidence/` — too big for git.
Only the scripts and the trimmed job log are kept here.

| file | what |
|---|---|
| `jax_dump.py` | instrumented JAX forward. Subclasses `NSB.DiffusionBlock` (must be a SUBCLASS — assigning a plain function to `__call__` loses Haiku's name scoping and the params fail to bind) and records 17 tensors per block with `jax.debug.callback`, which works inside `hk.scan`/`hk.remat`. `argv[5]` injects a fixture's `noised_*` so two runs share inputs exactly. Verified bit-exactly equal to the uninstrumented generator on both configs. |
| `torch_dump.py` | the same 17 tensors on the torch side (monkeypatches `blocks.DiffusionBlock.forward`). |
| `gpu_check.sbatch` | job 902051, h100, 3:30 — regenerates every fixture on GPU at default precision and at `JAX_DEFAULT_MATMUL_PRECISION=highest`. |
| `gpu_902051.trimmed.log` | that job's log with the XLA autotuner spam removed. |

In `/hnvme/workspace/v181bb10-salad/n005_evidence/`:

| file | what |
|---|---|
| `ORIG_model_{default_vp_scaled,multimotif_vp}.npz` | the two TF32 fixtures that caused the bug. Keep — they cannot be regenerated (GPU autotuning is not reproducible). |
| `regen_*`, `regen2_*` | CPU regenerations before the precision pin |
| `new_*` | CPU regenerations after the pin (bit-identical to `regen_*`, plus provenance keys) — these are what `fixtures/` now holds |
| `gpu_*`, `gpuhi_*` | H100, default precision vs `highest`. NOTE: `gpu_default_ve_scaled` and both `multimotif` arms were written after the generator was patched mid-job (generator mtime 22:24:59), so they already carry the pin; the clean pre-patch pair is `gpu_default_vp` (22:23:21) and `gpu_default_vp_scaled` (22:24:16). |
| `jaxdump_*`, `inj_*`, `torchdump_*` | the per-block bisection data |

## The numbers

GPU vs CPU, identical inputs and params (`out_pos`, max abs, Angstrom):

| config | H100 default | H100 `highest` |
|---|---|---|
| default_vp | 5.98e-03 | 7.63e-06 |
| default_vp_scaled | 5.95e-03 | 3.41e-05 |
| default_ve_scaled | 3.53e-03 | 4.48e-05 |

Torch vs CPU JAX, block 0, `default_vp_scaled` (the config that "failed"):
`current_neighbours`/`cond_neighbours` 0/3072 idx differ, `pair_1` 2.38e-06,
`local_norm` 4.17e-07, `block_out_pos` 2.10e-05, `final_pos` 9.27e-06.
