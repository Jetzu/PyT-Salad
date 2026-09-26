# N016 · Defensible lower bound on per-step time for exact and fast presets

type: research · date: 2026-09-26 · status: done

## Goal

Establish a DEFENSIBLE lower bound on the per-step time of both inference
presets at L=100 on an H100, and say how far each is from it. N009's
"3.4-3.5 ms hard floor" was refuted by N012 (it divided 2.096 ms of cuBLAS time
by 321 kernels and called the quotient a per-launch floor - an average, not a
bound). This item must not repeat that, and must label every term as a bound on
the PROBLEM or a bound on THIS IMPLEMENTATION.

NOTE: the brief called this item N017 and `results/N017-speed-floor/`, but
N017 was already taken by the concurrent Triton-audit item. This is N016 and
`results/N016-speed-floor/`.

## Plan

`T >= max( FLOPs/peak , HBM_bytes/BW , D * c_sync )`, with every term on the
right MEASURED on the node rather than estimated:

* `D` = critical-path depth of the traced aten graph counting only ops that
  provably force a global barrier (pointwise, views and gathers counted free -
  generous to the implementation, therefore safe for a lower bound)
* `c_sync` = marginal replay cost of an EMPTY kernel inside a captured CUDA
  graph, plus a hand-written grid barrier as the alternative primitive
* peaks and bandwidths measured, and every kernel timed INSIDE a cuda graph
  (an eager loop measures the ~6-8 us CPU dispatch, not the kernel)

## Changes

None - measurement and analysis only. All work in a detached worktree at
`/hnvme/workspace/v181bb10-salad/n016-floor` (HEAD a8533a7), because other
agents own blocks.py / diffusion.py / fast.py / opt.py / segment.py.

## Evidence

Placement: one H100 on the `h100` partition per job, all under 1 GPU-h, total
< 2.5 GPU-h. GPU was mandatory - the whole question is CUDA launch and
occupancy behaviour, which has no CPU analogue.

| job | what | result |
|---|---|---|
| 903966 | x_micro v1 | VOID: timed eager loops, i.e. CPU dispatch (~6-8 us), not kernels. Kept as the counterexample. |
| 903977 | x_micro v2 (graph-replayed) | c_launch = 1.166 us/kernel (empty, serial chain, flat 128-2048); 0.203 us with 8 forked streams; HBM 2.94 TB/s, L2 6.53 TB/s; GEMM at model shapes 3.1-16.2 us |
| 903979 / 903989 | grid barrier | 1.48 us (1 block) .. 1.88 us (132) .. 2.04 us (264) - MORE than a graph kernel launch, so a mega-kernel does not lower the floor |
| 903980 / 903996 / 904012 | x_census P/F/B/D | 10.014 GFLOP matmul + 0.348 GFLOP non-matmul; 30.1 MiB live params; **D = 205** critical-path barriers; weighted critical path 0.2581 ms fp32 / 0.2390 ms tf32 |
| 903971 | x_baseline | exact lean 3.573 ms = 3.395 replay + 0.178 wrapper; fast 3.007 = 2.835 + 0.172 |
| 903967 / 903990 | kernel census | exact 946 kernels 3.516 ms (cuBLAS 359 @ 6.02 us); fast 934 / 2.930 ms (cuBLAS 347 @ 4.57 us); no kernel below 1.2 us |
| 903996 | Triton vs cuBLAS | narrow projections 1.21-1.37x faster on Triton AND bit-identical at fp32; wide ones slower |
| 903998 / 904007 | concurrency | 8 independent forwards in one graph cost 3.6x one: **2.22x (exact) / 2.26x (fast)** at L=100, only 1.27x at L=400 |
| 904007 | D at L=400 | D = 205 exactly - the depth is length-independent; arithmetic takes over near L~208 |
| 904016 | min materialised traffic + ncu DRAM bytes | (red-team item R2) |

## Outcome

**exact >= 0.258 ms/step, fast >= 0.239 ms/step at L=100** (13.8x / 12.6x below
the measured 3.573 / 3.007 ms). The floor is a SERIALISATION floor: arithmetic
is 0.150 ms, non-matmul arithmetic 0.005 ms, irreducible HBM traffic 0.011 ms,
and the 205 unavoidable global barriers on the critical path are 0.24-0.26 ms.
The two presets have essentially the SAME per-step floor because every
relaxation `fast` is allowed attacks a term that was never binding; the two
diverge only above L~208, where fp32 arithmetic takes over (L=400: 0.599 vs
0.241 ms).

Second floor, same kernels: the **capacity floor** measured by saturating the
GPU with 8 concurrent forwards is ~1.66 ms (exact) / ~1.39 ms (fast). So the
gap splits in two - 3.57 -> 1.66 ms is scheduling (the GPU is 55% idle during
one forward at L=100), 1.66 -> 0.26 ms needs different kernels.

Largest unclaimed win, and it is not fusion: **2.2x more designs per second
today** by capturing 8 designs' steps in one graph, byte-identical computation.

Also corrected: N007's "the dist projection runs at 51 TFLOP/s, 76% of fp32
peak, nothing left inside it" - measured inside a graph it is 16.19 us =
20.2 TFLOP/s = 30% of peak. The 51 figure was eager-loop CPU dispatch.

## Outputs

`.agent-research/results/N016-speed-floor/RESULTS.md` (the derivation, the two
gap lists and a section naming what a red team can still break), plus
`x_micro.py`, `x_micro2.py`, `x_census.py`, `x_baseline.py`, `x_triton_gemm.py`,
`x_concurrency.py`, `x_ncu.py`, `run_*.sbatch`, `logs/`.
Worktree REMOVED (/hnvme is inode-capped and in grace). To reproduce, the
scripts hardcode `ROOT = /hnvme/workspace/v181bb10-salad/n016-floor`; recreate
it with

    git worktree add --detach /hnvme/workspace/v181bb10-salad/n016-floor a8533a7
    ln -sfn /home/hpc/v181bb/v181bb10/SALAD_torch/params  <that>/params
    ln -sfn /home/hpc/v181bb/v181bb10/SALAD_torch/fixtures <that>/fixtures

and `module load cuda/12.6.2` for the two jobs that JIT-build a CUDA extension
(M2/M6, the grid barrier) - nvcc is NOT in the triad env and /usr/local/cuda
does not exist on the compute nodes.

## Next

1. Batched-design sampler (2.2x throughput, zero numerics) - biggest item, not started.
2. Multi-stream capture of ONE forward's independent work (latency form of the same).
3. Per-shape Triton/cuBLAS selection for the narrow projections (~0.12 ms, bit-exact at fp32).
4. Capture N sampler steps per graph to remove the 0.178 ms wrapper.
5. R2 (traffic) is the one term a red team should attack first - see RESULTS.md section 10.
