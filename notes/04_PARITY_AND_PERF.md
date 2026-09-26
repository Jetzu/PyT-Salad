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

## 6. Reduced precision does NOT help — measured, do not retry

L=100, one forward, H100 (`scripts/bench_prec.py`, job 902105):

| mode | ms | 500-step | max abs dev vs fp32 |
|---|---|---|---|
| fp32 (`matmul="highest"`) | 7.03 | 3.52 s | - |
| fp32 + TF32 (`matmul="high"`) | 6.93 | 3.47 s | 0.55 A |
| autocast **bfloat16** | **19.85** | 9.93 s | 5.26 A |
| autocast **float16** | **18.88** | 9.44 s | 4.01 A |

TF32 IS available in PyTorch (`torch.set_float32_matmul_precision("high")`) and
buys ~1.5% here, against 20% for JAX. bf16/fp16 autocast are ~3x SLOWER.

Why, and why it is not surprising once the diagnosis is right: this model is
bound by KERNEL COUNT, not FLOPs (~837 Triton kernels/forward at 2.1 us each;
the GEMMs are only ~40% of the time and already run at ~52 TFLOP/s, near fp32
peak). Autocast does not reduce the number of kernels - it ADDS one cast per op
boundary - so it pays the launch cost twice over with no arithmetic saving to
recover it. The near-zero TF32 gain predicts exactly this.

Corollary for anyone optimising further: lower precision is the wrong axis.
The lever is FEWER, BIGGER kernels. And bf16 would cost parity too (4-5 A).

## 7. N009: the `exact` inference preset (2026-09-26)

Executing the N007 plan's ranked proposals under one rule: hold the parity
floor. All numbers are **one denoiser forward at L=100**, `default_vp-200k`,
H100, fp32 `matmul_precision="highest"`, manual whole-forward CUDA graph, one
length per process. `full` = a step that also runs the AA decoder + angle head
(what a *reporting* step costs); `lean` = `need_aa=False` (what the other 499
steps of a 500-step design cost).

### 7.1 The finding the plan did not have: `DotDict` breaks the Dynamo graph

`DotDict.__getattr__ = dict.get`. Dynamo's `_getattr_static` raises
`AttributeError` on that, gives up, and **breaks the graph at every
`config.<key>` read inside the compiled region**. `torch._dynamo.explain` on
`forward_core`:

| | graphs | breaks |
|---|---|---|
| before | 30 | 29 |
| after hoisting every hot config read into plain attributes | **1** | **0** |

Secondary cause in the same family: the `_slice(uniform, i)` helper was
compiled as its own frame and specialised on `i`, so 19 draws blew Dynamo's
default `cache_size_limit` of 8 and everything past the eighth broke. One
`torch.unbind` up front removes it.

The model was correct, "compiled", CUDA-graphed - and in thirty pieces.
`fast.py`'s own docstring claimed graph breaks were 0. **Rule: never read
`self.config` inside a `forward` on the compiled path. Hoist it in `__init__`,
and check with `torch._dynamo.explain` rather than by reading the code.**

### 7.2 The ladder (each rung adds to the one above)

| rung | what | full ms | lean ms |
|---|---|---|---|
| N007 baseline | manual CUDA graph, 30 Dynamo graphs | 4.966 / 4.941 | 4.489 / 4.485 |
| + **P0** | the graph-break fix above (bit-exact) | 4.886 / 4.870 | 4.006 / 4.015 |
| + **P3** | one-hot x matrix -> row gather (bit-exact) | 4.713 | 3.906 |
| + **P6** | hoist the loop-invariant neighbour stages (bit-exact) | 4.650 | 3.828 |
| + **P4+P5** | skip the dead pair-conditioning work, K 64->48 | **4.390** | **3.574** |

(four repeats of the last rung across two jobs: full 4.390 / 4.384 / 4.365 /
4.385, lean 3.574 / 3.569 / 3.556 / 3.564.)

P0 is worth almost nothing on a full step and **0.48 ms on a lean one** - the
broken graph was costing most of its fusion in the trunk, which is all a lean
step runs. P2 is the full->lean column itself: **0.82 ms** at the shipping
config (it was 0.48 ms before P0).

End to end, `generate.py --num_aa 100 --fast`: **1.5 s per design**, from
~2.1 s. `--preset exact` is the default; `--preset fast` is N011's.

Other lengths, shipping config (two independent jobs agree):

| L | full ms | lean ms | N007 baseline (full) |
|---|---|---|---|
| 50 | 3.738 / 3.724 | 3.029 / 3.020 | 4.22 |
| 100 | 4.390 / 4.384 / 4.365 / 4.385 | 3.574 / 3.569 / 3.556 / 3.564 | 4.95 |

The gain is roughly flat in absolute terms, as the kernel-count diagnosis
predicts: what was removed is launches, and a launch costs the same at L=50 as
at L=100.

### 7.3 P1 was already subsumed, and is now moot

The angle head's CPU sidechain tables made *inductor* refuse to cudagraph that
partition. The manual whole-forward capture does not care, which is why the
launch gap was already closed at 4.95 ms - and after P2 the angle head does not
execute at all on 400 of a run's 401 steps. Not implemented; no longer worth an
hour.

### 7.4 `need_aux=False` is still a dead end - re-measured post-P0

4.378 vs 4.385 ms full, 3.572 vs 3.564 lean. The dssp head and the distogram
neighbour draw cost nothing measurable even with the graph unbroken. Leave
them on.

### 7.5 THE CUDA DENOISER IS NOT REPRODUCIBLE - read before any PDB diff

Two runs of the same command with the same `--seed` produce **different
proteins**. Measured with the control that matters (job 903106: `allon` vs
`allon2`, identical flags, identical seed - DIFFERS on all three designs).

Source, measured (job 903136):

```
index_put_ stage chaining reproducible over 20 reps : True
selected neighbour indices reproducible             : True  (0 differing slots)
index_mean reproducible                             : False  max|d| 1.639e-07
```

Bug B1 (section 4) was the prime suspect and is **refuted** - it reproduces
exactly. The source is `index_sum`/`index_mean` -> `scatter_add`, whose CUDA
atomicAdd ordering varies at 1.6e-7, twice per block, six blocks deep. And it
does not stay at 1.6e-7: each block's `pos` feeds the next block's neighbour
selection, and a stable argsort over near-ties flips *discretely* under a 1e-7
nudge. One forward, run twice, identical input:

```
NOISE FLOOR: the identical call run twice differs by 1.755e-02 on cuda
                                                     0.000e+00 on cpu
```

**The CUDA forward's own run-to-run noise is ~900x the 2e-5 CPU parity floor.**
Consequences, in order of importance:

1. A trajectory diff, a PDB diff or an absolute GPU tolerance measures the
   backend, not the change. Assert against an identical-input rerun.
2. The parity fixtures being CPU-generated (section 5) is not just about TF32;
   it is the only reason the 2e-5 floor is measurable at all.
3. `--seed` does not pin a CUDA `--fast` run. Fixing that means a deterministic
   segment reduction, which needs its own parity re-baseline.

### 7.6 What each proposal is worth numerically, on the gate that works

CPU is deterministic, so the single forward there is the real test, and there
every proposal is **bit-identical**:

| | CPU | CUDA, vs the 1.755e-02 noise floor |
|---|---|---|
| P2 `need_aa=False`, 1 chain | `torch.equal` True | 1.514e-02 = 0.86x |
| P2 `need_aa=False`, 2 chains | `torch.equal` True | 1.307e-02 = 0.74x |
| P4 dmap/flags terms dropped | `torch.equal` True | 0.000e+00 |
| P4+P5 `pair_cond=False`, K 64->48 | `torch.equal` True | 1.611e-02 = 0.92x |

P3 and P6 live in the shared modules, so `tests/test_model_parity.py` covers
them directly; it did not move (0.00002 / 0.00020 A on all four configs).

P4/P5 carry a **runtime gate**: `FastDenoiser` derives "this design has no pair
conditioning" from the real `dmap_mask`/`flags`, never from the config. Taking
the skip when `flags` IS set moves `pos` by 2.1e+02 A - `flags` channel 0 is
"different chains", so anything multi-chain, any template dmap and any explicit
`chain_contacts` switches it off by itself.

### 7.7 Inductor kernel-merging knobs: one small win, one hard crash

| knob | full ms | lean ms | |
|---|---|---|---|
| (none) | 4.385 | 3.564 | shipping |
| `aggressive_fusion=1` | 4.302 | 3.503 | -1.9 % / -1.7 % |
| `combo_kernels=1` | — | — | **crashes** |

`combo_kernels` (horizontal fusion - exactly the "merge independent launches"
lever this model wants) raises in inductor codegen on the stable-argsort node:

```
Error in codegen for ComputedBuffer(... Sort(... stable=True ...))
```

torch 2.5.1. Not available to us while neighbour selection is a stable argsort,
which it must be (section 1).

`aggressive_fusion` is real but small, and it is a PROCESS-GLOBAL inductor flag
whose numerical effect cannot be resolved on CUDA - the backend's own noise
floor (section 7.5) is 1.755e-02, three orders of magnitude above anything a
fusion decision could move. **Not shipped in `exact`.** If someone wants it,
the gate is a bit-equality check of the compiled forward on CPU, where the
floor is 0.

### 7.8 The floor, named (kernel census, L=100, `x_kernels.py`)

The unit of account is kernels removed. Measured, profiling the same compiled
callable with `capture=False` (a captured graph profiles as one opaque op):

| | kernels | GPU-busy |
|---|---|---|
| **every proposal OFF**, reporting step | **1349** | 4.872 ms |
| shipping, reporting step | 1202 | 4.351 ms |
| **shipping, lean step** | **946** | **3.521 ms** |

**403 kernels removed**, 4.872 -> 3.521 ms. Where they still go:

| | kernels | GPU-busy | |
|---|---|---|---|
| reporting step (`need_aa=True`) | 1202 | 4.351 ms | |
| — GEMM (cuBLAS) | 419 | 2.616 ms | **6.24 us each** |
| — Triton (fused pointwise/reduction) | 645 | 1.519 ms | 2.36 us each |
| — sort / argsort | 57 | 0.107 ms | 1.87 us each |
| — other | 79 | 0.103 ms | 1.31 us each |
| lean step (`need_aa=False`) | **946** | **3.521 ms** | |
| — GEMM (cuBLAS) | 321 | 2.096 ms | 6.53 us each |
| — Triton | 524 | 1.269 ms | 2.42 us each |
| — other + DtoD copies | 101 | 0.157 ms | |

The lean step's individual top kernels:

```
triton_ (all fused pointwise/reduction, incl. the stable argsorts)  524x  1.269 ms
sm80_xmma_gemm_f32f32..._nn_n_tilesize32x32x8                       120x  0.838 ms
sm80_xmma_gemm_f32f32..._nn_n_tilesize64x64x8                        24x  0.270 ms
sm80_xmma_gemm_f32f32..._tt_n_tilesize64x64x8                        18x  0.269 ms
sm80_xmma_gemm_f32f32..._tt_n_tilesize32x32x8                        31x  0.194 ms
cutlass_80_simt_sgemm_64x64_8x5_nn_align1                            30x  0.137 ms
gemv2N_kernel                                                        24x  0.078 ms
Memcpy DtoD                                                          57x  0.073 ms
splitKreduce_kernel                                                  38x  0.070 ms
```

(The argsorts do not appear separately: inductor lowers `torch.argsort(
stable=True)` to a Triton `Sort` node, so they are inside the 524.)

**60 % of a lean step is 321 cuBLAS launches at 6.5 us each, against a 0.21 ms
arithmetic floor.** N007 section 2 already measured why: the two wide
projections (`dist` 400->64, `pairvec` 240->64) run at ~51 TFLOP/s, ~76 % of
the H100's fp32 FFMA peak, and the narrow ones (`dir` 60, `rot` 9) cost the
*same* 6.4 us because they are pure launch latency. There is nothing left
inside them.

So **2.0 ms is not reachable on the exact path.** Getting there needs ~1.5 ms
off the GEMM stage, i.e. ~240 of the 321 launches gone. The two candidates:

* **P7** (batch the condition branch's six blocks into one `bmm`) removes at
  most 30 of them, ~0.19 ms, for 2-3 days - and a `bmm(batch=6)` may pick a
  different tile or split-K, which spends parity budget for 5 % of a step.
* **P8** (one Triton kernel per pair block, features computed in registers)
  is the only change that removes launches by the hundred, and the N007 plan
  classifies it "materially different" numerics by construction: it reassociates
  an 842-term dot product into whatever its tiling does. That is the `fast`
  preset's licence, not this one's.

The honest exact floor is **~3.4-3.5 ms** for a lean step, ~1.5 s for a
100-residue design.
