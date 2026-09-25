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

## 4. Red-team findings (2026-09-25)

Two blockers made the generation path unreachable; both fixed.

| # | finding | status |
|---|---|---|
| A1 | `apply_diffusion` passed `noise=` to `diffuse_sequence`, whose kwarg is `uniform=` — `TypeError` on **every** call, so the whole noise->denoise entry point was dead. The parity tests missed it because they feed `pos_noised` straight from the fixture. | fixed |
| A2 | `make_initial_data` emitted int32 indices; torch `scatter_add` demands int64, so the first segment op aborted before the trunk. | fixed (coerced inside the segment ops, so any caller is safe) |
| A3 | segment ops raised on out-of-range indices where JAX silently drops the write and clamps the read (non-dense chain ids). | fixed, matches JAX exactly |
| A4 | no working float64 path: `LinearToPoints` hard-cast to fp32. | fixed |
| B1 | the `-1` sentinel un-masks residue N-1 and it gets selected twice | **NOT fixed — upstream-faithful, see below** |

### B1: batched inference is not result-preserving (upstream bug)

In `get_neighbours`, the stage chaining writes `+inf` over already-chosen
neighbours. Rows containing `-1` write too, and `-1` indexes column N-1 with
the *original finite* distance — undoing the `+inf` a genuine neighbour N-1
just received. Residue N-1 is then re-selected in the next stage and
double-counted in attention.

Measured consequences (L=6 protein, fp32):

| | max deviation |
|---|---|
| same protein duplicated in a 2-item batch, half 1 vs half 2 | `pos` 3.46 A |
| two proteins batched vs run separately | `pos` 2.34 A, 3/24 aatype flipped |
| L=6 alone vs L=6 + 4 masked padding residues | `pos` 4.94 A on the *real* residues |

Confirmed deterministic, not stochastic: re-running the batch test with the
uniform field TILED so both positions read identical random numbers gives
byte-identical results to an untiled field (batch item 0 vs item 1 = 5.844 A
either way; standalone vs +4 masked padding = 6.731 A). A stochastic effect
would have moved. The divergence comes from the deterministic index/spatial
chaining, not the Gumbel draw.

This is NOT the ordinary "batch>1 gives different samples" property of a
diffusion model - that is expected and desirable. This is the same input
producing different output depending on its offset in the flat residue axis.

**Verified bit-identical in JAX on the same input** — the port is faithful and
salad is wrong. So: *a protein run as anything other than the last item of the
flat residue axis, or run beside masked padding, gets a different answer than it
gets standalone.* Batch inference and padded loaders are not result-preserving.
Do not "fix" this without regenerating every fixture; it would break parity.

Latent GPU risk: both sides rely on unspecified duplicate-scatter ordering, and
torch's `index_put_` with duplicate indices is documented non-deterministic on
CUDA. Untested — no GPU was reachable during the audit.

### Clean under attack

Degenerate sizes (L=1..64), all-masked / single-unmasked inputs, collinear
backbone atoms, 1e8 coordinates, NaN confinement, non-contiguous tensors, bf16
autocast, thread-count determinism, and input mutation across the whole public
surface: no defects. Gradients are clean — no NaN/Inf, `vec_norm`'s
clamp-before-sqrt does protect the backward pass (grad exactly 0 at 0), and both
zero-init layers receive gradient.

**Training note:** 202 parameters get `grad=None` — all of them the `.b` biases
that Haiku allocates but never applies. Harmless for a normal optimizer, but
**DDP aborts without `find_unused_parameters=True`**.

New parity regimes checked and passing (worst deviation on unmasked rows):
multi-batch + multi-chain + padding `pos` 7.3e-6; full motif/dssp/aa
conditioning `pos` 9.1e-6; degenerate chain lengths `pos` 3.9e-6; all 21 aatypes
through `single_protein_sidechains` 2.0e-7.

## 5. RESOLVED: the two "failing" configs were TF32 fixtures

| config | max coord deviation | aatype |
|---|---|---|
| default_vp | 0.00002 A | 0/48 differ |
| default_ve_scaled | 0.00002 A | 0/48 differ |
| default_vp_scaled | 0.00002 A | 0/48 differ |
| multimotif_vp | 0.00020 A | 0/80 differ |

All four are now at the fp32 floor. The earlier 0.00589 A / 0.03694 A were
**not** a port defect: `fixtures/model_default_vp_scaled.npz` and
`fixtures/model_multimotif_vp.npz` were generated on an H100
(`.agent-research/results/N004-porting-gaps/gen_fixtures_task1.sbatch`,
`--partition=h100 --gres=gpu:h100:1`) while the two passing fixtures were
generated on the CPU login node. **XLA:GPU lowers float32 `dot` to TF32
(10 mantissa bits) by default**, so those two fixtures are a ~1e-3-relative
reference, not an fp32 one.

Measured, same inputs, same params, same container (N005):

| run | out_pos vs CPU | traj block 0 | aatype |
|---|---|---|---|
| default_vp on H100, default precision | 5.98e-03 | 1.22e-02 | 0 differ |
| default_vp on H100, `highest` | 7.63e-06 | 2.38e-05 | 0 differ |
| default_vp_scaled on H100, default | 5.95e-03 | 1.38e-02 | 0 differ |
| default_vp_scaled on H100, `highest` | 3.41e-05 | 2.21e-04 | 0 differ |
| default_ve_scaled on H100, default | 3.53e-03 | 1.86e-03 | 0 differ |

`default_vp` - the config that always passed - reproduces the whole "failure"
signature when its fixture is regenerated on a GPU. The gap was never a
property of `vp_scaled`/`multimotif`, only of which machine wrote the fixture.
Two further facts from the same experiment:

- Two H100 runs of the same config at default precision differ from **each
  other** by 4.47e-03 A (XLA autotuning picks different dot kernels), so a
  GPU-generated fixture is not even self-reproducible.
- The port itself was always correct. Dumping every DiffusionBlock
  intermediate on both sides against a CPU JAX run with the stored fixture's
  `noised_*` injected (`.agent-research/results/N005-parity-bisect/`), torch
  matched to 9.27e-06 A in `final_pos` and <=2.1e-05 in every block-0 tensor -
  the same floor as `default_vp`. The neighbour index arrays
  (`current_neighbours`, `cond_neighbours`) matched exactly, 0/3072 per block.
- The trunk is contractive, not chaotic: a 1e-5 perturbation at
  `prepare_features` output leaves ~1e-5 at `out_pos`. There is no
  amplification to hide behind.

**Guards now in place**

- `scripts/gen_fixtures_model.py` sets
  `jax.config.update("jax_default_matmul_precision", "highest")` before any
  computation. Verified a bit-exact no-op on CPU (all four fixtures byte-for-byte
  unchanged) and it collapses the H100 gap by ~800x.
- Every fixture now records `jax_platform` and `jax_matmul_precision`.
- `tests/test_model_parity.py` refuses any fixture whose recorded precision is
  not `highest`, and any fixture predating the pin. Verified: pointed at the old
  H100 file it exits 1 with an explanatory message instead of reporting a
  300x parity failure.

Debugging note for the next person: `aatype` matching exactly is **not**
evidence that a numeric gap is small. A TF32 reference keeps the argmax intact
while moving coordinates by 6e-03 A. Check fixture provenance before
bisecting the network.
