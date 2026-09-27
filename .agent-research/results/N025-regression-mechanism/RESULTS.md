# N025 — where in the rollout it goes wrong, and what state is responsible

Branch `n006-manual-cudagraph`. Known-good `c5ab7a0`, HEAD-under-test `fd9ba32`.
Two detached worktrees under `/hnvme/workspace/v181bb10-salad/`
(`n025_good` = c5ab7a0, `n025` = fd9ba32). All instrumentation is OUTSIDE the
repo, in `/hnvme/workspace/v181bb10-salad/n025_x/` (copied into `scripts/`).

Every number below is at L=100, seed 42, and — where two arms are compared —
on the SAME PINNED SCHEDULE (`--num_steps 250 --out_steps 200`), so the
`fast` preset's `steps_scale=0.5` is never a free variable.

## 0. The regression, reproduced a third time

Job 904678 (another agent's 5-arm bisect) DIED ON THE /hnvme INODE QUOTA after
82 of the 160 designs of its second arm; arms C/D/E never ran and its
`bisect_designs/c_noaggr`, `d_nop7`, `e_both_off` are empty. Its two surviving
arms are still valid data:

| arm | n | diverged | converged CA-CA | converged Rg |
|---|---|---|---|---|
| c5ab7a0 `--fast --preset fast` | 160 | **0** | 3.7605 | 13.07 |
| fd9ba32 `--fast --preset fast` | 82 | **13 (15.9 %)** | **3.8729** | **21.62** |

Read the CONVERGED columns, not the diverged count: the bond length of the
SURVIVORS is up 0.11 A and their Rg is up 8.5 A. This is a distribution shift,
not a tail.

**It is not cross-design state accumulation.** Binned by design index, HEAD's
first twenty designs are already at CA-CA 3.8633 / Rg 21.68 and the last are at
3.8511 / 20.49 — flat. The defect is present in design 0.

## 1. The model CODE is bitwise innocent (CPU, deterministic by construction)

`x_cpu_bits.py`, job 904718 (cpu partition, 48 cores). `forward_core` run
EAGERLY on CPU from dumped, identical inputs, at L=64 and L=100,
`pair_cond=False` (the unconditional de-novo gate), all eleven outputs:

| comparison | max abs diff | bitwise equal |
|---|---|---|
| c5ab7a0 vs fd9ba32 (defaults) | **0.000000e+00** | yes, 11/11, both lengths |
| c5ab7a0 vs fd9ba32 `SALAD_OPT_BATCHCOND=0` | **0.000000e+00** | yes, 11/11, both lengths |

So every model-code change in `c5ab7a0..fd9ba32` — P7's batched condition
prologue, `det_segment_sum`, the `cond_pair` plumbing — is a **no-op in this
configuration when executed eagerly**. Whatever broke is in what the GPU
actually RUNS, not in what the model computes.

The only remaining differences on that path are the three things c5ab7a0 never
touched at all (its `FastDenoiser.__init__` sets no inductor config whatsoever,
only `set_float32_matmul_precision`):

* `torch._inductor.config.aggressive_fusion = True`
* `torch._inductor.config.joint_custom_post_pass = fx_graph_cse` (N017/O1, the
  HEAD commit itself)
* `opt.BATCH_COND` / P7 — bitwise-free on CPU, but a `bmm(batch=6)` instead of
  six `mm`s on the GPU

(placeholder — remaining sections filled from jobs 904715 / 904716 / 904722 /
904739 / 904745)
