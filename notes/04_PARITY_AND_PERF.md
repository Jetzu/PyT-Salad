# Parity and performance: measured results

All numbers 2026-09-25, `default_vp-200k`, H100 for timings.

## 1. Numerical parity

| level | test | result |
|---|---|---|
| geometry functions (38) | `tests/test_geometry_parity.py` | exact, worst 4e-6 |
| 6-layer trunk, real weights | `tests/test_stack_parity.py` | rel 2.2e-6, **0.0003 A** max coord |
| whole denoiser, real weights | `tests/test_model_parity.py` | rel <=1.2e-5, **0.00002 A** max coord, aatype 0/48 differ |
| 81-step sampling trajectory | `tests/test_sampling_parity.py` | see below - chaotic |

The checkpoint loads with `load_state_dict(strict=True)`: 779 tensors, 0 missing,
0 unexpected, no transposes, no shape fudging.

### The one real bug found

`jax.nn.gelu` is declared `def gelu(x, approximate: bool = True)` - the **tanh**
approximation - and every salad call site takes that default.
`torch.nn.functional.gelu` defaults to `approximate='none'` (exact erf).

Measured cost of getting this wrong, over the 6-layer trunk in ONE denoiser call:

| | trunk local | trunk pos | final coords |
|---|---|---|---|
| erf (wrong) | rel 4.1e-3 | rel 3.2e-3 | 0.043 A mean, **0.31 A max** |
| tanh (right) | rel 4.8e-6 | rel 2.2e-6 | 0.0000 A mean, 0.0003 A max |

The sampler runs 400-500 such calls. This is the class of defect that only
numerical parity testing catches - the model still "works" with the wrong
nonlinearity, it just quietly produces different proteins.

## 2. Sampling is chaotic - read this before trusting a trajectory diff

A single forward pass matches JAX to 2e-7 relative. A *trajectory* does not,
and cannot: the loop feeds its own output back in, so fp32 reduction-order
differences are amplified exponentially at high noise, then contract as the
schedule anneals.

Measured on L=64, 81 steps, all randomness pinned and replayed:

| | worst on trajectory | final mean | final max | aatype differ |
|---|---|---|---|---|
| same torch code, ONE input scalar perturbed by 1e-6 | 0.583 A | 0.0065 A | 0.023 A | 0/64 |
| torch vs JAX | 4.37 A | 0.054 A | 0.351 A | 1/64 |

Cross-backend divergence is ~7x the intrinsic noise floor on the worst step and
~8x on the final mean - the same order of magnitude. Both backends produce a
real protein with identical CA-CA statistics (3.762 A; native ~3.8).

**Consequence:** assert trajectory agreement as a RATIO to an identical-code
rerun, never as an absolute tolerance. `tests/test_sampling_parity.py` does this.

## 3. Performance (H100, single denoiser call, median of 20)

| L | JAX (jit) | Torch eager | Torch compiled | 500-step design: JAX / compiled |
|---|---|---|---|---|
| 50 | 3.65 ms | 74.5 ms | 32.9 ms | 1.8 s / 16.5 s |
| 100 | 3.96 ms | 74.5 ms | 33.1 ms | 2.0 s / 16.5 s |
| 200 | 5.21 ms | 74.7 ms | 33.0 ms | 2.6 s / 16.5 s |
| 400 | 7.72 ms | 74.9 ms | 33.3 ms | 3.9 s / 16.7 s |
| 800 | 14.30 ms | 75.0 ms | 67.6 ms | 7.2 s / 33.8 s |

Eager torch time is **flat from L=50 to L=800**, which is the signature of being
launch-bound: the cost is Python dispatch and kernel launches, not arithmetic.
JAX is jitted into one fused graph; the port is not.

`torch.compile` on the DiffusionBlocks gives 2.2x (74 -> 33 ms). It cannot be
applied to the whole model: Dynamo does not trace the python-dict shuffling in
`StructureDiffusionPredict.prepare_condition` faithfully and drops the
`condition` key.

Remaining performance work, in expected order of payoff:
1. **Compile the neighbour selection.** Each block does two dense `(N,N)`
   distance + stable-argsort passes, 12 per forward, and they currently sit
   *outside* the compiled region (in the `neighbour_fn` closure).
2. **CUDA graphs** (`mode="reduce-overhead"`) currently fails with
   "accessing tensor output of CUDAGraphs that has been overwritten" at
   `blocks.py` `Rigid.apply_to_point` - the block outputs must be cloned.
3. Restructure the model's forward to take tensors rather than dicts so the
   whole thing can be compiled as one graph.

Correctness was the goal for this pass; none of the above changes numerics.
