# N007 — Restructuring the SALAD diffusion block to match JAX speed

Research plan, 2026-09-25. **No production code was written.** All numbers below
are measured on an H100 (partition `h100`, fp32 `matmul_precision=highest`,
`torch.compile(mode="reduce-overhead")`, CUDA graphs engaged, `default_vp-200k`,
L=100 unless stated). Measurement scripts and raw logs live beside this file:
`x_micro.py`, `x_pairsplit.py`, `x_batch6.py`, `x_e2e.py`, `logs/`.

Reference points: **torch 6.5–7.0 ms**, **JAX 4.73 ms fp32** (3.95 ms at its
TF32 default). Target: ≤ 4.73 ms without touching the released checkpoints or
the 0.00002 Å parity.

---

## 1. Verdict up front

The gap is closable, but **not by any of the fusion ideas in the brief.** Pair-
linear fusion and the flash-IPA recast were both implemented and measured, and
both are *slower* than what the port does today. What is left is a different
class of change: stop launching kernels that produce nothing.

| | |
|---|---|
| Measured GPU-busy time / forward | **5.375 ms** in ~1840 kernel launches (mean 2.9 µs) |
| Measured wall time / forward | 6.47–7.03 ms → **~1.1–1.6 ms is launch gap**, i.e. work that is *not* on the GPU |
| Arithmetic floor (14.1 GFLOP at 67 TFLOP/s fp32) | 0.21 ms |
| Memory floor (~1.1 GB of traffic at 3 TB/s) | ~0.37 ms |

The model runs at **~13× its own hardware floor**. It is bound by *kernel
count* against a ~2–3 µs per-launch floor, exactly as the brief suspected — but
the consequence is that the correct unit of account for every proposal is
**kernels removed**, not FLOPs saved or bytes saved. A change that removes 100
launches buys ~0.25 ms whatever it does to the arithmetic; a change that saves
30 % of the FLOPs but adds one copy kernel per site buys nothing. That is
precisely why the fusion experiments failed.

Ranked proposals (§4) total an estimated **−2.0 to −2.5 ms**, landing at
**≈4.2–4.6 ms**, i.e. at or just under JAX fp32. Three of the six are
bit-exact; two perturb summation order only; none require a checkpoint change,
and only one (P7) touches the converter at all — and that one's weight
transformation is the *identity* (it stops un-stacking what Haiku already
stacked).

---

## 2. Where the time actually goes (measured, L=100)

Component timings are isolated benchmarks: the real 12 pair blocks / 12 IPAs /
6 neighbour selections lifted out of the model, given real weights and real
inputs, each compiled and CUDA-graphed on its own.

| component | ms | % of 6.5 ms | kernels |
|---|---|---|---|
| pair features ×12 (build + 7 GEMMs + LN + 2-layer MLP) | **2.03** | 31 % | 363 (30.2 per pair block) |
| — of which: the 7 projections + LN + MLP | 1.31 | 20 % | 156 GEMM launches |
| — of which: feature construction | 0.72 | 11 % | 187 Triton |
| sparse IPA ×12 | **1.35** | 21 % | — |
| neighbour selection, all 6 blocks (both selectors) | **0.23** | 3.5 % | — |
| everything else (prepare_features, 6× Update + update_positions, AA decoder ×3, angle head) + launch gap | ~2.9 | 45 % | — |

Two things in that table are worth reading twice.

**(a) 72 % of the pair-block GPU time is cuBLAS SGEMM, and most of those GEMMs
do no work.** The 7 projections per pair block are `(6400, in_i) @ (in_i, 64)`.
Only two of them are large: `dist` (in=400) runs at ~51 TFLOP/s — genuinely near
fp32 peak — and `pairvec` (in=240). The other five (`relpos` 66, `dmap` 64,
`flags` 3, `dir` 60, `rot` 9) each cost the *same* ~6.4 µs as the 400-wide one
because they are entirely launch- and latency-bound. Dropping just three of them
(`relpos`, `dmap`, `flags`) takes the projection stage from 1.307 ms to
**1.051 ms** — measured, −0.256 ms, −12.6 % of the whole pair stage.

**(b) `torch.compile` is already doing the frame CSE.** Built in isolation,
`direction_features` costs 0.313 ms/12 and `position_rotation_features`
0.296 ms/12, dominated by `aa_frames_t` in each; built together with the other
five features the *total* is 0.719 ms, not the 1.285 ms the parts sum to.
Inductor is sharing the Gram–Schmidt. So "hoist the frames" is not a proposal —
it is already done, and re-plumbing it by hand would gain nothing.

### 2.1 Two structural dead zones found by inspection + measurement

These are the two findings that most of §4 rests on.

**The block-contact neighbour stage returns nothing.** `extract_condition_
neighbours(16, 16, 16, 16)` chains four stages; the fourth ranks by
`pair_condition["flags"]`, and with no flags its distance matrix is `+inf`
everywhere, so every selected index is the `-1` sentinel. Measured:

```
L=100: cond_neighbours (100,64)  -1 entries 1600/6400
   index 0/1600   spatial 0/1600   random 0/1600   block_contact 1600/1600
L=400: same, 6400/25600, all in block_contact
```

**25 % of every condition-branch pair feature and 25 % of every condition-branch
IPA key is computed on slots that are masked to exactly zero.** Not
approximately: the logits are set to `-1e9`, `exp(-1e9 - max)` underflows to
exactly `0.0`, and the post-softmax re-mask sets them to `0.0` again, so their
contribution to `out_pair`/`out_scalar`/`out_point` is `0.0 * finite = 0.0`.

**No duplicate neighbour slots occur at L ≥ 100.** I checked for the B1
double-selection (notes/04 §4) because it would forbid recasting sparse
attention as dense masked attention: measured 0 duplicates in both selectors at
L=100 and L=400. It can still occur when a row contains `-1` (short chains,
padded batches), so any dense recast would have to be gated on L — see §5.3.

---

## 3. The constraint, stated precisely

At L=100 every kernel in this model is far below the size at which the GPU
starts caring what it computes. The pair stage's mean kernel is 5.25 µs; the
whole forward's is 2.9 µs. A null kernel on an H100 costs ~1.5–2.5 µs to
schedule even from inside a CUDA graph.

So the arithmetic of the plan is simple:

```
t ≈ N_kernels × 2.9 µs + launch_gap
6.5 ms ≈ 1840 × 2.9 µs + 1.15 ms
```

To reach 4.73 ms we need either ~570 fewer launches, or the launch gap closed
plus ~200 fewer launches. Both are available.

**Corollary, and it is the most important sentence in this document:** any
restructuring that *adds* a materialisation — a `cat`, a staging buffer, a
transpose — pays ~2.5 µs per site plus its traffic, and must remove at least one
whole kernel per site to break even. Inductor will not fuse a `cat` into
producers that are gathers or reductions (all of ours are), so it emits one copy
kernel per input. This is the complete explanation of §5.1 and §5.2.

---

## 4. Proposals, ranked by expected gain per unit of risk

| # | proposal | est. gain | effort | parity | converter |
|---|---|---|---|---|---|
| P1 | Make the sidechain tables device-resident so the angle-head partition is CUDA-graphed | **≤ 1.0 ms** | 1 h | **bit-exact** | untouched |
| P2 | Skip the AA decoder + angle head on non-output sampling steps | **~0.5–0.9 ms** | 4 h | **bit-exact in `pos`** | untouched |
| P3 | One-hot × matrix → row gather (`relpos`, `linear_aa`) | ~0.15 ms | 2 h | **bit-exact** | untouched |
| P4 | Skip the `dmap`/`flags` pair terms when there is no pair conditioning | ~0.25 ms | 3 h | **bit-exact** | untouched |
| P5 | Drop the all-`-1` block-contact stage: K 64→48 on the condition branch | ~0.2–0.4 ms | 3 h | summation order only | untouched |
| P6 | Hoist the loop-invariant neighbour stages out of the 6-block loop | 0.07 ms @L=100, 0.24 ms @L=400 | 3 h | **bit-exact (verified)** | untouched |
| P7 | Batch the whole condition branch's pair features across the 6 blocks into one `bmm` prologue | see §4.7 | 2–3 d | summation order only | *identity* transform |
| P8 | Hand-written Triton kernel fusing feature construction + projection | see §4.8 | 1–2 w | materially different | non-persistent buffer |

### P1 — CUDA-graph the angle-head partition  (≤1.0 ms, bit-exact, 1 h)

**What.** `geometry/sidechain.py` keeps `restype_rigid_group_default_frame` &c
as module-level **CPU** tensors and moves them to the device lazily inside
`_tables(device)`. When Dynamo first traces `torsion_angles_to_frames`, the
cache is cold, so the CPU tensor is lifted into the graph as an input and
inductor refuses to CUDA-graph that partition. The compile log says so
verbatim:

```
skipping cudagraphs due to cpu device (arg5_1). Found from :
  decoders.py:338 single_protein_sidechains -> sidechain.py:351 _tables
```

Everything downstream of `single_protein_sidechains` therefore launches
kernel-by-kernel from the host. That is where most of the measured 1.1–1.6 ms
of launch gap lives.

**Fix.** Register the four tables as non-persistent CUDA buffers on `AngleHead`
(or pre-populate `_DEVICE_CACHE` at model `.to(device)` time) so no CPU tensor
ever enters a traced graph.

**Reasoning for the number.** GPU-busy is 5.375 ms against a 6.47–7.03 ms wall.
A fully graphed forward should show a gap of ~0.1–0.2 ms (the pair-stage
microbenchmark, which *is* fully graphed, shows 2.026 ms wall against 1.906 ms
busy — a 0.12 ms gap over 363 kernels). Recovering the gap on the remaining
~1500 kernels is worth up to ~1 ms. Treat ≤1.0 ms as the ceiling and ~0.5 ms as
the conservative estimate; the `x_e2e.py tables` run measures it exactly.

**Parity.** Bit-exact — the tensors are identical, only their residency changes.

**Converter.** Untouched. These are constants, not checkpoint parameters.

**Verify.** `TORCH_LOGS=+inductor` must no longer print `skipping cudagraphs`;
`tests/test_model_parity.py` unchanged at 0.00002 Å; `x_e2e.py tables` reports
the wall-time delta and `max|dpos|` vs unpatched eager (must be `0.000e+00`).

### P2 — Skip the AA decoder + angle head on non-output steps  (~0.5–0.9 ms)

**What.** `forward_core` always runs the 3-block AA decoder, the dssp head and
the angle head. The sampler (`diffusion/sampling.py`) writes `data["seq"]`,
`data["aatype"]` and `data["atom_pos"]` from the result every step — but the
*model* never reads them back: `aa_gt` is a base key that `scripts/generate.py`
explicitly documents as "never changes over the run", and the trunk's inputs are
`(pos_noised, prev_pos, prev_local, condition, aa_gt)`. On a step that is not in
`--out_steps`, every one of those three heads is dead.

**Fix.** Give `forward_core` a `predict_aa: bool` that is a *Dynamo constant*
(so it specialises into two graphs, both CUDA-graphed), and have the sampler
pass `True` only on output steps and on the final step. `FastDenoiser` keeps two
compiled callables and two static-input sets.

**Reasoning for the number.** Eager profiling (`outputs/prof_899930.log`) put the
AA decoder at 11.28 ms of a 75.2 ms eager forward (15.0 %) and the angle head at
1.35 ms (1.8 %). Those fractions are launch-count fractions, which is exactly
what carries over to the compiled regime: ~17 % of 5.4 ms ≈ 0.9 ms. The AA
decoder's 3 blocks run at K=32, so call it 0.5–0.9 ms. `x_e2e.py noaa` measures
it.

**Risk.** Behavioural, not numerical: if a future flag makes the sampler consume
`update["aa"]` mid-trajectory (`--sample_final` already re-runs the decoder on
out-steps; `diffuse_sequence` would), the flag must be forced on. Make
`predict_aa=True` the default and require the sampler to opt out explicitly, so
the failure mode is "slow", never "wrong".

**Parity.** Bit-exact for `pos`, `local` and `trajectory` — the skipped heads
are pure sinks. `aa`/`aatype`/`atom_pos` are only produced on the steps that
report them, which is the only place they were ever read.

**Converter.** Untouched — the weights still load, they are just not executed
every step.

**Verify.** Run a full 500-step design with and without the flag; `pos` at every
out-step must be **bit-identical**, not merely close (this is a skip, not a
reassociation, so `torch.equal` is the right assertion). Plus the existing
`tests/test_sampling_parity.py`.

### P3 — One-hot × matrix → row gather  (~0.15 ms, bit-exact)

**What.** `sequence_relative_position(32, one_hot=True)` materialises an
`(N, K, 66)` one-hot tensor (1.7 MB at L=100) and feeds it to a `(66, 64)`
Linear. That is a gather written as a GEMM. Same for
`AADecoderPairFeatures.linear_aa(one_hot(aa, 21))` and
`AADecoder.linear_aa_embed`.

**Fix.** Return the integer bins and index the weight: `W[binned]`.

**Why it is bit-exact.** Each row of the product is `Σ_k onehot[k] · W[k, :]`
with 65 terms exactly `0.0` and one term `W[idx, :]`. In IEEE-754, `0.0 · x = 0.0`
for finite `x` and `a + 0.0 = a` exactly, in any order. The only observable
difference is the sign of a zero, which no downstream op distinguishes. This
holds for cuBLAS's FMA accumulation as well.

**Reasoning for the number.** Removes, per site: one ~6.4 µs GEMM launch and one
one-hot materialisation kernel (1.7 MB write + 1.7 MB read). 12 diffusion pair
blocks + 3 AA-decoder blocks + the distogram decoder ≈ 16 sites × ~8 µs ≈
0.13 ms, plus the `linear_aa` sites. The measured upper bound is contained in
the 0.256 ms that dropping `relpos`+`dmap`+`flags` saved on the projection side.

**Converter.** Untouched. The parameter stays `linear_relpos.w` with shape
`(66, 64)` and loads with `strict=True` exactly as now; only `forward` changes.
Keep the `HkLinear` module and add a `forward_indexed()` rather than replacing
the class, so the state-dict key set is byte-identical.

**Verify.** `torch.equal` between the old and new pair tensors on real weights
and real neighbour indices, for every bin including the cross-chain bin `2c+1`;
then `tests/test_stack_parity.py` and `tests/test_model_parity.py` unchanged.

### P4 — Skip `dmap`/`flags` when there is no pair conditioning  (~0.25 ms, bit-exact)

**What.** For de-novo generation `dmap_mask` is all-false and `flags` is
all-false, so `torch.where(dmap_mask, distance_rbf(dmap), 0)` is an exactly-zero
`(N, K, 64)` tensor and `torch.where(flags.any(-1), flags, 0)` an exactly-zero
`(N, K, 3)`. Both are then multiplied by a weight matrix and added: two GEMMs,
two gathers, two `where`s and two adds per pair block, contributing exactly
`0.0`.

**Fix.** A boolean set **once per design** (not per step) on the model — e.g.
`Diffusion.set_pair_conditioning(active: bool)` — consulted by
`DiffusionPairFeatures.forward`. It must be a Python attribute so Dynamo
specialises on it; it must *not* be a config key, because config drives the
converter's index derivation (`diffusion_block_map`).

**Reasoning for the number.** The measured projection-side cost of the three
small linears `relpos + dmap + flags` is 0.256 ms across 12 pair blocks
(1.307 → 1.051 ms). `relpos` is P3's; `dmap + flags` is ~0.17 ms of GEMM, plus
their feature construction (a `(N,K)` gather, a 64-bin RBF over it, a `where`,
and an `any`) and two adds — call the total ~0.25 ms.

**Why it is bit-exact.** `0.0 @ W = 0.0` exactly (all products are `±0.0`, all
partial sums `0.0`), and `pair + 0.0 == pair` exactly. Verified by the
`x_e2e.py nopc` run reporting `max|dpos|` against the unpatched eager model.

**Converter.** **Untouched, and this matters.** `linear_dmap` and `linear_flags`
must stay in the module tree so `load_state_dict(strict=True)` still finds
`linear_1`/`linear_2` (and so `multimotif_vp`'s index derivation is unaffected).
Deleting the submodules would break the 779-tensor strict load. Skip the *call*,
never the *parameter*.

**Verify.** `x_e2e.py nopc` `max|dpos| == 0.0`; plus a regression that runs
`multimotif_vp` and a motif-scaffolding job with the flag **on** and confirms it
is never silently taken (assert the flag is False whenever `dmap_mask.any()`).

### P5 — Drop the all-`-1` block-contact stage: K 64 → 48  (~0.2–0.4 ms)

**What.** Given §2.1, when `flags` is all-false the fourth stage of
`extract_condition_neighbours` contributes 16 columns that are every one `-1`.
Skipping the stage shrinks `cond_neighbours` to `(N, 48)`, which shrinks the
*entire* condition branch by 25 %: `pair_2`'s seven feature tensors and seven
GEMMs, its LayerNorm and MLP, and `attn_2`'s gathers (`k[nbr]`, `k_g[nbr]`,
`v[nbr]`, `v_g[nbr]`), logits, softmax and three output einsums.

**Fix.** Same runtime flag as P4 (they are the same condition), passed to
`extract_condition_neighbours` as `num_block=0` — and, better, short-circuiting
the stage entirely rather than calling `get_random_neighbours(0)`, which still
computes an `(N, N)` Gumbel field before discarding it.

**Reasoning for the number.** The condition branch is 6 of 12 pair blocks
(≈1.0 ms) and 6 of 12 IPAs (≈0.68 ms). These kernels are latency-bound, not
size-bound, so a 25 % size cut does *not* buy 25 % of the time; expect 15–25 %
on the memory-bound portion and ~0 on the launch-bound portion, plus the whole
fourth stage's ~6 launches × 6 blocks. Estimate 0.2–0.4 ms at L=100 and
proportionally more at L=400/800 where the kernels stop being latency-bound.
`x_e2e.py k48` measures it.

**Parity.** Not bit-exact, but the deviation is *pure reassociation*: the dropped
slots contribute exactly `0.0` to every reduction, so the only change is the
tile/tree shape of the reductions over K. Expect ~1e-7 relative. The
`x_e2e.py k48` run reports the measured `max|dpos|`; the gate is that it stays at
or below the existing 2e-5 fp32 parity floor.

**Converter.** Untouched (neighbour selection holds no weights).

**Verify.** `x_e2e.py k48` `max|dpos|` ≤ 2e-5; assert at runtime that
`(cond_neighbours[:, 48:] == -1).all()` in a debug build; and a guard that the
flag is off whenever `flags.any()` (hotspots, chain contacts, multimotif).

### P6 — Hoist the loop-invariant neighbour stages  (0.07 ms @L=100, 0.24 ms @L=400, bit-exact)

**What.** The 6-block loop calls two selectors per block, 12 dense `(N, N)`
passes in total. Several stages cannot change between blocks:

* `get_index_neighbours(16)(resi, chain, batch, mask)` depends on nothing that
  the loop mutates. It is computed **twice per block, twelve times per forward**,
  and all twelve results are the same tensor.
* `prev_pos` is a loop invariant, so the *whole* of
  `extract_condition_neighbours`' distance matrix, its index stage and its
  spatial stage are invariant. Only the two Gumbel stages differ (different
  `uniform` slice per block).
* Inside `extract_neighbours`, `get_spatial_neighbours` computes
  `pair_distance(ca)` and then `get_random_neighbours` computes the *same*
  `pair_distance(ca)` again, because the latter is called with points rather
  than a matrix.

**Fix.** Compute the invariants once in the stack prologue and pass them in;
pass the already-built `(N, N)` CA distance matrix to the random stage (the
selector already accepts a square matrix — that overload exists upstream).

**Measured** (`x_micro.py neigh`):

```
L=100   v0 current 0.228 ms   v1 hoisted+shared 0.158 ms   -30.8%   bit-identical indices: True
L=400   v0 current 0.846 ms   v1 hoisted+shared 0.610 ms   -27.9%   bit-identical indices: True
```

**Parity.** Bit-exact, and *verified* bit-exact: the stacked index tensors
compare `torch.equal == True`. This is pure common-subexpression elimination —
no reassociation, since the removed work is literally re-executed identical
arithmetic.

**Converter.** Untouched.

**Verify.** `torch.equal` on the neighbour index tensors (already in
`x_micro.py neigh`); then `tests/test_model_parity.py` — the N005 bisect
established that the index arrays match JAX 0/3072 per block, so any change here
shows up immediately.

**Honest note on priority.** notes/04 lists "compile the neighbour selection" as
remaining work item #1. It has since been done, and the whole of neighbour
selection is now **3.5 % of the forward**. P6 is cheap and free of risk, but it
is not where the remaining gap is. The note should be corrected.

### P7 — Batch the condition branch across the 6 blocks

**What (the structural idea the brief asked for).** `pair_2` in block *b*
computes its features from `prev_pos` — the *same tensor for all six blocks* —
gathered at `cond_neighbours_b`. Nothing in that computation depends on
`features`, on `pos`, or on any previous block's output. Only the weights differ
per block. The whole condition-branch feature stage is therefore a **prologue**,
and a prologue over six independent, identically-shaped problems is one batched
problem:

```
today   6 blocks × (≈12 feature kernels + 7 GEMMs + LN + 2 MLP GEMMs + adds)
        ≈ 174 launches
batched ≈12 feature kernels on 6× data + 7 bmm(batch=6) + LN + 2 bmm + adds
        ≈ 29 launches
```

Implementation shape: tile `prev_pos` six times along the flat residue axis (the
port already uses a flat, padding-free residue layout, and `chain`/`batch`
offsets keep the copies from seeing each other), stack the six neighbour index
sets with a per-copy offset — mapping `-1` to `(b+1)·N − 1` reproduces NumPy's
negative-index wrap *within* the copy, bit-exactly — run the existing,
unmodified feature builders once, then `bmm` with stacked weights.

**The converter consequence is the interesting part, and it is favourable.**
Haiku's `hk.LayerStack` already stores every trunk parameter with a **leading
axis of `diffusion_depth`**: `…/diffusion_block/linear_7/w` is literally
`(6, 66, 64)` in the checkpoint. Today `build_state_dict` does `arr = arr[layer]`
to un-stack it into six modules. P7's weight transformation is therefore *not* a
concatenation or a permutation — it is **the identity, minus the slicing step**.
`load_diffusion_stack` keeps the same `diffusion_block_map` table and the same
config-dependent index derivation; it simply loads the stacked array into one
`StackedPairFeatures` parameter instead of six sliced ones.

What *does* change is the **torch** state-dict key set:
`diffusion_stack.blocks.{0..5}.pair_2.linear_dist.w` collapses to
`diffusion_stack.cond_pair.linear_dist.w`. Any `.pt` saved from the port needs a
rename shim; the Haiku `.npz`/`.jax` checkpoints — the ones that must keep
loading — are unaffected and in fact map more directly than before.

**Expected gain.** Being measured by `x_batch6.py`; the kernel-count argument
predicts 145 launches removed ≈ 0.42 ms at the 2.9 µs floor, partially offset by
the batched kernels being 6× larger (they are still small: the 6-way `dist`
bmm is 6 × 327 MFLOP ≈ 2.0 GFLOP, 38 µs at 51 TFLOP/s, versus 6 × 6.4 = 38 µs
today — a wash on arithmetic, all the gain on launches). Peak extra memory:
`(6, N, K, 400)` fp32 = 61 MB at L=100, 246 MB at L=400.

**Parity.** Summation order only. A `bmm(batch=6)` may pick different tiles or a
split-K where six separate `mm`s did not. Gate on ≤2e-5.

**Effort.** 2–3 days. It is the only proposal that changes the module tree, and
it needs a matching change in `MotifAdapterDiffusionBlock` (whose `pair_2` runs
over `pos`, *not* `prev_pos`, so **it is not batchable** — the motif block must
keep the sequential path; the code has to support both).

**Verify.** `torch.equal` on the six stacked pair tensors is too strong (bmm
reassociates); assert `max|d| ≤ 2e-5` against the sequential path on real
weights, then `tests/test_model_parity.py` for all four configs — including
`multimotif_vp`, which must take the *un*batched branch.

### P8 — A hand-written Triton pair-feature kernel (the big hammer)

**What.** One kernel per pair block that streams `pos[neighbours]` from HBM once
and, inside its K-loop, computes the RBFs / directions / rotations / pair vectors
on the fly and accumulates straight into the `(N, K, 64)` output — never
materialising the 21.5 MB of `(N, K, 842)` intermediates at all.

**Why it is the only fusion that can work.** §5.1 shows that every *framework-
level* fusion loses because the concatenation is an extra 43 MB round trip and
~7 extra copy launches, which exactly cancels the ~5 saved GEMM launches. A
Triton kernel does not concatenate: the features exist only in registers.

**Expected gain.** Replaces ~30 launches per pair block with ~1–2. If it lands
at the traffic floor (read 1.5 MB of gathered positions, write 1.6 MB) it is
~3 µs per pair block versus ~168 µs today. Realistically 20–40 µs, i.e.
**1.2–1.6 ms saved across 12 pair blocks.** This is by far the largest single
number in the plan.

**Why it is ranked last anyway.** It changes the accumulation order of a 842-term
dot product to whatever the kernel's tiling does, it must reproduce
`vec_norm`'s clamp-before-sqrt and the exact `gaussian_rbf` denominator
(`step²`, not `2σ²`) and the endpoint-inclusive vs midpoint bin-centre
distinction (`linear_centers` vs `_arange_centers` — the port documents both and
they differ), and any of those getting silently wrong is exactly the class of
defect that produced the 0.31 Å `gelu` bug. Budget 1–2 weeks including the
parity harness, and do it only after P1–P7 have been measured.

**Converter.** The kernel wants one contiguous `(842, 64)` weight. Build it as a
**non-persistent buffer** in a `load_state_dict` post-hook from the individual
`linear_*.w` parameters, which stay in the module tree. That keeps
`strict=True`, keeps `diffusion_block_map` untouched, and keeps the port
trainable (the buffer is rebuilt from the parameters, so gradients still flow to
the parameters — or, for training, fall back to the un-fused path).

---

## 5. Refuted — measured, not argued

### 5.1 Fusing the 7 pair projections (brief item 1) — **dead, all three ways**

```
pair-feature block ×12, L=100, K=64, concatenated width 842
  v0 current (7 GEMM + 6 add)          2.019 ms
  v1 runtime cat of feats AND weights  2.056 ms   -1.8%   max|d| 1.62e-05
  v2 LOAD-TIME concatenated weight     2.158 ms   -6.9%   max|d| 1.62e-05
  v3 v2 + features written directly
     into one pre-allocated buffer     2.057 ms   -1.9%   max|d| 1.62e-05
```

All three are **slower** than the status quo, and the load-time variant — the one
the brief specifically asked about — is the *worst* of the three.

**Why.** The arithmetic says fusion should win: five of the seven GEMMs are pure
latency at ~6.4 µs each, so collapsing them should save ~32 µs per pair block
against a fused GEMM that costs the same as the two large ones it absorbs. It
does not win because inductor emits **one copy kernel per concatenated input**
(it will not fuse a `cat` whose producers are gathers and reductions), so the
concatenation costs ~7 launches plus a 43 MB read-write round trip ≈ 31 µs —
cancelling the saving almost exactly. Writing the features directly into a
pre-allocated buffer (v3) does not help either: inductor lowers the slice
assignment to the same copy kernels.

Note also the **1.62e-05 deviation**: the fused version is not free
numerically. It reassociates a 7-way sum of separate 64/66/240/400-term dot
products into one 842-term dot product, landing at the same order as the current
whole-model JAX parity (2e-5). It would have to be re-baselined even if it were
faster.

**Conclusion for the brief's question 1:** load-time fusion does *not* do better
than runtime fusion — it does worse. Fusing the feature construction too does
not rescue it. The only fusion that can win is one that never materialises the
concatenated tensor (P8).

### 5.2 The flash-IPA recast (brief item 4) — **dead: slower and less accurate**

I implemented `flash_ipa`'s algebra for the sparse case exactly as upstream
derives it (query/key components concatenated so that one inner product yields
`Q·K/√d + 2Q_p·K_p − Q_p² − K_p²`, using the port's own
`dfactor = softplus(γ)·w_C/2` rather than upstream's `overscale_dfactor`, which
notes/02 correctly flags as *not* reproducing standard IPA):

```
sparse IPA ×12, L=100, K=64
  v0 current              1.351 ms
  v1 flash-IPA algebra    1.379 ms   -2.1%   max|d| 1.373e-04
```

**Why it does not help.** The algebra converts one broadcast-subtract-square-sum
(which inductor already fuses into a *single* Triton kernel) into a gather of a
wider key tensor plus one `einsum`. Launch count is unchanged; the gathered key
grows from `(N,K,H,32) + (N,K,H,8,3)` = 2.8 M elements to `(N,K,H,58)` = 3.0 M.
There is nothing to win: the dense-attention framing that makes flash-IPA a
large win in *dense* IPA (it lets you call a fused SDPA kernel) has no sparse
analogue, because each row has its own 64 keys and no SDPA kernel takes a
per-row key gather.

**And the fidelity cost is real.** `Σ(q−k)²` → `Σq² − 2Σq·k + Σk²` is
catastrophic cancellation waiting to happen: `|q_g|²` is ~7200 Å² for a protein
20 Å off origin while the difference term is ~9 Å², so the fp32 relative error is
amplified by ~800×. The measured 1.37e-04 is **7× the port's whole-model parity
floor** and, unlike the other deviations in this document, it *scales with the
distance of the structure from the coordinate origin* — i.e. it is a latent
failure mode that L=100 centred test cases do not expose.

### 5.3 Recasting sparse attention as dense masked attention — **blocked by an upstream bug**

At L ≤ 128 the neighbourhood (K=64) is over half the protein, so dense
`(N, N, H)` attention would be only ~1.5× the arithmetic while eliminating every
`[neighbours]` gather — attractive on paper. It is **not equivalent**, because
of the B1 double-selection documented in notes/04 §4: a row containing `-1`
un-masks residue `N−1`, which can then be selected twice, and softmax over a
list with a duplicate gives that residue double weight, while a dense masked
softmax gives it single weight. I measured 0 duplicates at L=100 and L=400, so
the recast *would* be safe there — but it silently changes results for short
chains, for padded loaders and for the batched case, and notes/04 is explicit
that this bug must not be "fixed" without regenerating every fixture. Not worth
a conditional fast path.

### 5.4 TF32 — off the table, and small anyway

`matmul_precision="high"` measures 6.47 → 5.94 ms (−8 %), and `max_autotune`
6.11 ms. But TF32 is 10 mantissa bits, and N005 established that TF32 moves
coordinates by ~6e-3 Å while leaving `aatype` intact — i.e. it looks fine and is
not. The JAX comparison point is 4.73 ms *at matched fp32* precisely because of
that finding. Do not re-open this.

### 5.5 Things already done that notes/04 still lists as to-do

* "Compile the neighbour selection" — done; neighbours are now 3.5 % of the
  forward, not the #1 item.
* "CUDA graphs fail at `Rigid.apply_to_point`" — fixed by `aa_frames_t`/`apply_t`
  on bare tensors; graph breaks are 0. **But** P1 shows one partition is still
  not graphed, for an unrelated reason.
* "Hoist `aa_frames_t`" — inductor already CSEs it (§2b). Not a proposal.

---

## 6. The floor, and what it would cost to go below it

Adding the estimates: P1 (0.5–1.0) + P2 (0.5–0.9) + P3 (0.15) + P4 (0.25) +
P5 (0.2–0.4) + P6 (0.07) ≈ **1.7–2.8 ms**, from 6.5–7.0 ms to **≈4.2–5.0 ms**.
P7 adds perhaps another 0.3–0.4 ms. So JAX fp32 (4.73 ms) is reachable with
bit-exact and reassociation-only changes; **beating** it comfortably needs P7,
and beating it by a wide margin needs P8.

Below that there is a hard floor, and it is not arithmetic:

* After P1–P7 the forward still launches on the order of **1000–1200 kernels**.
  At the H100's ~2–2.5 µs floor for a graph-replayed kernel that is
  **2.5–3.0 ms**, against an arithmetic floor of 0.21 ms and a bandwidth floor of
  0.37 ms. Everything between 0.4 ms and 2.5 ms is launch overhead that only
  kernel *merging* (P8, or a full hand-written block) can remove.
* The two large GEMMs per pair block (`dist` 400→64, `pairvec` 240→64) are
  already at ~51 TFLOP/s, i.e. ~76 % of the H100's fp32 FFMA peak. **There is
  nothing left in them.** The only way to make them faster is tensor cores,
  which means TF32/bf16, which means abandoning parity (§5.4).
* The 6 trunk blocks are strictly sequential (each block's `pos` feeds the next
  block's neighbour selection), and within a block the two attention branches are
  chained through the residual stream. There is no concurrency to exploit beyond
  the condition-branch prologue of P7.

**So: yes, the remaining gap can be closed without changing numerics — but only
just, and only by deleting work rather than by reorganising it.** Going
materially *below* JAX means either P8's custom kernel (accumulation-order
change, ~1–2 weeks) or reduced precision (checkpoint-invalidating in effect, and
already ruled out).

---

## 7. Sequencing

| step | work | gate |
|---|---|---|
| 1 | P1 + P3 + P6 (all bit-exact, ~6 h total) | `torch.equal` on the affected tensors; `tests/test_model_parity.py` **unchanged at 0.00002 Å**; re-measure wall time |
| 2 | P4 + P5 behind one `pair_conditioning_active` runtime flag | `max|dpos|` = 0 for P4, ≤2e-5 for P5; a guard that asserts the flag is off whenever `dmap_mask.any() or flags.any()`; `multimotif_vp` parity must not move |
| 3 | P2, default-off, sampler opts in | 500-step design: `pos` at every out-step **bit-identical** to the current sampler |
| 4 | re-measure against JAX; stop here if ≤4.73 ms | |
| 5 | P7 (module-tree change + converter simplification + torch-state-dict rename shim) | all four configs at ≤2e-5; `multimotif_vp` takes the sequential branch |
| 6 | P8 only if 4 and 5 leave a gap worth a fortnight | dedicated bit-level harness per feature builder before the kernel is wired in |

## 8. Standing rules for this work

1. **Never delete a parameter to skip a computation.** Skip the call, keep the
   module; `load_state_dict(strict=True)` over 779 tensors is the port's main
   safety net and the converter's index derivation
   (`_pair_feature_names` → `linear_14` / `_18` / `_20`) depends on the module
   set matching the config.
2. **Any derived weight (concatenated, stacked, transposed) is a
   non-persistent buffer rebuilt in a post-load hook**, never a parameter. The
   state-dict key set must stay byte-identical unless the change is P7, which
   documents its rename.
3. **Classify every change up front** as bit-exact / reassociation-only /
   materially different, and assert accordingly: `torch.equal` for the first,
   ≤2e-5 for the second, a re-baselined fixture for the third. The measurements
   above show the three classes really do separate (0.0 vs 1.6e-5 vs 1.4e-4).
4. **Re-measure the whole forward, not the piece.** Every fusion experiment here
   looked good on paper and lost end to end.
5. **Fixtures stay CPU-generated.** N005: an H100-generated JAX fixture is TF32
   and is not even self-reproducible (4.47e-3 Å between two runs).

---

## Appendix — measurements and how to reproduce

All under `.agent-research/results/N007-block-restructure/`:

| script | what |
|---|---|
| `x_micro.py diag` | neighbour-stage `-1` census and duplicate-slot check |
| `x_micro.py pairfuse` | the three fusion variants of §5.1 |
| `x_micro.py neigh` | P6, with a `torch.equal` bit-exactness assertion |
| `x_micro.py ipa` | the flash-IPA recast of §5.2 |
| `x_pairsplit.py` | feature-construction vs projection split, per-feature costs, kernel census |
| `x_batch6.py` | P7, the batched condition-branch prologue |
| `x_e2e.py <variant>` | end-to-end wall time + deviation for each patch, monkeypatched (no production code) |
| `run*.sbatch` | h100, 1 GPU, ≤55 min each (≈0.9 GPU-h per job) |

Raw logs in `logs/`.
