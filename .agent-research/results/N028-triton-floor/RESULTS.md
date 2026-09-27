# N028 - the Triton launch floor: O2's verdict, the single-op kernels, and the 3x3 bmm

Target: the ~250-350 us of removable Triton launch floor N017 identified and
found no safe route to. **Found 829 us at L=100** - 3.389 -> 2.560 ms lean
(-24.5 %) - of which **-5.0 % is bit-exact and lands in `exact`**.

Baseline, everywhere below: worktree detached at `98bc61b` (P7 off), L=100,
`default_vp`, preset `exact`, lean step (`need_aa=False`, `pair_cond=False`),
H100, `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`, one length + one arm per process.
**Lean step 3.389 ms, 782 kernels = 392 Triton (1060 us) + 352 cuBLAS
(2138 us) + 38 memcpy, 3244 us GPU busy.**

## Summary

| # | change | file | lean ms | vs base | class | ships |
|---|---|---|---|---|---|---|
| - | base (98bc61b) | - | 3.389 | - | - | - |
| **O2** | IPA einsum -> mul+sum (N017's, finished here) | `modules/attention.py` | 3.292 | **-2.9 %** | reassociation | **`fast`** |
| **O6** | `direction_features` is a SLICE of `pair_vector_features` | `geometry/features.py` | 3.218 | **-5.0 %** | **bit-exact** | **`exact` + `fast`** |
| **O8** | every 3x3 frame transform is a cuBLAS bmm | `geometry/frames.py`, `rigid.py` | 2.773 | **-18.3 %** | reassociation | **`fast`** |
| | O2+O6 | | 3.123 | -7.8 % | | |
| | O6+O8 | | 2.695 | -20.5 % | | |
| | **O2+O6+O8** | | **2.560** | **-24.5 %** | | the proposed `fast` |

Every arm is 2 reps, one process each; replicates agree to <=0.003 ms.

## 1. O2's n=160 verdict: PASSES. It ships in `fast`.

N017 left O2 as "a measured candidate, not a recommendation": -3.1 % but
`test_sampling_parity` fails at 7/64 residues, and no structural validation had
been run. Run here at N015's n=160 per arm (job 907123, L=100, `--preset fast`,
seeds 42 n=40 + 1234 n=120 pooled, `tests/test_design_quality.py`, three arms
in three processes off a frozen tree):

|arm|n|CA-CA (A)|Rg (A)|helix|strand|coil|clash|diverged|
|---|---|---|---|---|---|---|---|---|
|base|160|3.7634+-0.0123|12.79+-1.49|0.592|0.325|0.083|0.19|**0/160**|
|O2  |160|3.7623+-0.0128|12.76+-1.46|0.577|0.337|0.086|0.20|**0/160**|
|O6  |160|3.7634+-0.0129|12.81+-1.62|0.603|0.317|0.081|0.16|**0/160**|

* Welch vs base: every metric **p >= 0.43** (O2), **p >= 0.59** (O6).
* Two-sample KS: every metric **p >= 0.21** (O2), **p >= 0.40** (O6).
* Brown-Forsythe on the spread: nothing significant, let alone Bonferroni.
* Rg tail counts (mean +- 3 sd of base): 3 / 3 / 3 - identical.
* Fisher on diverged: 0/160 vs 0/160, p=1.0.
* **Power**: at n=160 this detects 0.096 in helix, 0.086 in strand, 0.46 A in
  Rg, 0.0039 A in CA-CA. Largest observed: 0.015 / 0.012 / 0.03 / 0.0011. The
  null is a null *with power*, which is the whole point of n=160 (N015 got the
  same question wrong twice at n=8 and n=40).
* FIREDCHECK: 0/160 PDBs byte-identical to base in either arm, so both flags
  demonstrably ran (the "identical to three decimals" trap).

So the answer the brief asked for: **O2's structures hold up, and it ships in
`fast`.** It cannot ship in `exact` - `test_sampling_parity` fails at 25.2x the
noise floor (job 907165), the same signature N017 measured and the same class
as `SALAD_OPT_DETSEG=1`.

## 2. Why 215 Triton kernels carry exactly one op

`ir_pre/post_fusion` + `TORCH_LOGS=fusion` + the captured-replay profile, all
from one arm of job 907103. The counts below reconcile exactly with the
profiler, which is the check that matters: a fused group's members are printed
in `ir_post_fusion.txt` as bare `opN: SchedulerNode` lines exactly like real
singles, and counting those naively gives 729 "singles" instead of 215.

**734 pointwise/reduction scheduler nodes -> 177 fused groups (mean 2.93) +
215 single nodes = 392 Triton kernels, plus 296 extern (cuBLAS) nodes.**
So **55 % of Triton launches carry one node** (63 % pre-O1; O1's CSE improved it).

### 2a. The reasons, ranked

105,394 `cannot fuse A with B: <reason>` decisions, 8 distinct reasons:

| decisions | reason | of the 215 singles, how many ever hit it |
|---|---|---|
| 48,193 | `intermediate nodes between node1 & node2` | 182 (85 %) |
| 20,350 | `no shared data` | 152 (71 %) |
| 18,394 | `will increase peak memory` | 135 (63 %) |
| 9,640 | **`node1/node2 is extern or nop`** | **181 (84 %)** |
| 6,835 | `numel/rnumel mismatch` | 101 (47 %) |
| 1,814 | `memory deps did not match` | 112 (52 %) |
| 150 | `nodes numel incompatibility` | 23 (11 %) |
| 18 | `tiling mismatch` | 12 (6 %) |

Zero of the 215 were never considered, and zero were blocked *only* by the
extern rule - every single node hits several reasons, so no one reason is "the"
cause. **The structural finding is the 84 % row: a cuBLAS call is a FUSION
BARRIER as well as a 5.98 us launch.** The graph alternates
elementwise -> bmm -> elementwise -> bmm, and inductor cannot fuse across an
extern kernel, so each GEMM removed is worth its launch *plus* the fusions it
was blocking. That is why O2 and O8 pay far more than their launch arithmetic
predicts (O8 removes 72 cuBLAS launches = 431 us of launch, and takes 622 us
off the step).

### 2b. What the single-op kernels actually are

44 distinct kernels whose whole body is one ATen op: 149 launches, **409 us =
39 % of all Triton time**.

| launches | us | us/launch | aten op |
|---|---|---|---|
| 35 | 128.3 | 3.67 | **`aten.clone`** |
| 34 | 120.7 | 3.55 | **`aten.cat`** |
| 19 | 61.9 | 3.26 | `aten.sort` (the stable argsort) |
| 20 | 31.4 | 1.57 | `aten.add` |
| 12 | 25.5 | 2.12 | `aten._softmax` |
| 15 | 19.1 | 1.27 | `aten.sub` |
| 8 | 12.1 | 1.51 | `aten.stack` |
| 5 | 8.6 | 1.72 | `aten.index` |

**`clone` + `cat` = 249 us = 23 % of all Triton time, and both are pure data
movement.** Every `clone` is a layout copy to get a permuted operand into
`bmm` layout - i.e. the cuBLAS calls are paying for themselves twice. The most
expensive single Triton kernel in the whole step is `triton_poi_fused_clone_20`,
**6 launches x 11.30 us = 67.8 us of nothing but laying out `k[neighbours]` for
`attention.py:115`'s einsum**.

By innermost source line (top of 392 launches / 1060 us):

| launches | us | source |
|---|---|---|
| 19 | 107.0 | `attention.py:115` the IPA `dot` einsum -> **O2** |
| 48 | 101.4 | `basic.py:96` `out = out + self.b` (HkLinear bias) - N027's |
| 12 | 88.2 | `features.py:214` `pair_vector_features`' final `cat` |
| 33 | 84.5 | `basic.py:147` LayerNorm - N027's |
| 19 | 61.9 | the stable argsorts |
| 13 | 59.1 | `neighbours.py:72` `knn = cat((neighbours, knn))` |
| 12 | 49.2 | `features.py:210` `both = cat([self_pos, neigh_pos])` -> **O6** |
| 7 | 46.7 | `attention.py:109` the IPA `dist` reduction |
| 12 | 46.4 | `features.py:56` the distance-RBF grid |
| 24 | 41.4 | `attention.py:121` softmax |
| 20 | 25.3 | `frames.py:70` `aa_frames_t` |
| 12 | 19.8 | `features.py:196` `position_rotation_features`' `@` -> **O8** |
| 10 | 19.2 | `frames.py:86` `apply_t`'s einsum -> **O8** |

## 3. O6 - `direction_features` is a slice of `pair_vector_features`. BIT-EXACT, -5.0 %.

`DiffusionPairFeatures.forward` calls `direction_features(pos, nb)` and
`pair_vector_features(pos, nb)` on the SAME arguments. Both gather
`pos[neighbours]`, push it through `apply_inverse_t(aa_frames_t(pos))` and
`vec_normalized` it; the pair-vector version just does it for
`cat([self_pos, neigh_pos])` first. **The first function's entire output is the
trailing `A` rows of the second's intermediate.** So the neighbour gather, the
frame transform and the normalisation all run twice per pair block.

`fx_graph_cse` (N017/O1) cannot see it: the `cat` makes the argument of
`apply_inverse_t` a structurally different node, so the two subgraphs are not
syntactically equal even though one computes a superset of the other. **This is
the class of redundancy a graph CSE cannot reach and only a source change can.**

`geometry/features.direction_and_pair_vector_features` computes both in one
pass; `SALAD_OPT_PAIRFEAT=0` restores the two calls.

**Measured -5.0 %** (3.389 -> 3.218 ms, 2 reps each): 782 -> 759 kernels,
Triton 392 -> 381 (1060 -> 1039 us), cuBLAS 352 -> 340, GPU busy 3244 -> 3070 us.
Length sweep: L=50 2.853 -> 2.730 (-4.3 %), L=100 -5.0 %, L=200/400 below.

**Bit-exactness.** The shared ops are elementwise per 3-vector, so no reduction
order changes, and the controls agree:

* function level, fp32 and fp64: `torch.equal` True for both outputs;
* whole model, eager CPU, seeded: bitwise True;
* **whole compiled `forward_core`, CPU (deterministic by construction), all
  eleven outputs, L=64 AND L=100, in TWO independent jobs on two different
  trees (907143, 907159): `bit_identical=True`, max|d| = 0.0 exactly.**

That is the same evidence standard O1 (`fx_graph_cse`) shipped on.

**Gates.** Full parity suite `SUITE_EXIT=0` (job 907165): default_vp 0.00001 A,
default_vp_scaled 0.00001, default_ve_scaled 0.00002, multimotif_vp 0.00015,
geometry 38/38, stack, encoder, sampling all pass - identical to the flags-off
control run in the same job. n=160 design quality **0/160 diverged**, all
statistics above. Compiled-vs-eager on the GPU (K021's missing check, job
907161): pos 1.07x the flags-off floor at L=64 and 0.94x at L=100, aatype 0/L
differ - nothing resembling P7's 5-orders-of-magnitude blowup.

**PORTABLE.** Four lines of ordinary PyTorch in `geometry/features.py`. No
hand-written Triton, no tuned constant, no arch guard - inductor generates and
autotunes the kernels per device exactly as before.

## 4. O8 - every 3x3 frame transform is a cuBLAS launch. -18.3 %, reassociation.

`apply_t`, `apply_inverse_t`, `rot_apply`, `rot_apply_inverse` and `rot_compose`
are written as `torch.einsum("...ij,...j->...i", ...)` (and `@`). **ATen lowers
every single one of them to `aten.bmm`.** Verified two ways: `make_fx` on the
bare functions emits `permute, unsqueeze, permute, permute, view, permute, view,
bmm, view, permute, view`; and **61 of the 303 `extern_kernels.bmm` calls in
the generated module carry a `(..., 3, 3)` operand.**

So a third of this model's cuBLAS launches are 3x3 frame algebra: a 5.98 us
launch for nine multiply-adds, plus the `clone` inductor emits to get the
permuted operands into bmm layout, plus the fusion barrier of section 2a.

Written as broadcast-multiply + `sum` over the contracted axis
(`SALAD_OPT_FRAMEMULSUM=1`), each becomes an ordinary pointwise+reduction node
that fuses into its neighbours:

| | lean ms | kernels | Triton | cuBLAS | GPU busy |
|---|---|---|---|---|---|
| base | 3.395 | 782 | 392 / 1062 us | 352 / 2139 us | 3248 us |
| **O8** | **2.773 (-18.3 %)** | 723 | 405 / 1089 us | **280 / 1480 us** | 2615 us |
| O6+O8 | 2.695 (-20.5 %) | 702 | 384 / 1010 us | 280 / 1480 us | 2536 us |
| O2+O6+O8 | **2.560 (-24.5 %)** | 635 | 401 | **232** | 2419 us |

-72 cuBLAS launches for +13 Triton ones, and 659 us off the GEMM bill.

**It is NOT bit-exact and cannot enter `exact`.** It is a reassociation over a
3-element contraction (cuBLAS is free to contract with FMA in a different
order). Bitwise at the function level and in eager CPU per-function tests, but:

* whole compiled `forward_core`, CPU: 9 of 11 outputs differ, L=64 and L=100;
* `test_model_parity` **FAILS** `default_ve_scaled` on `atom_pos`
  (rel 2.216e-05 against a 2e-05 tolerance - a marginal miss, `pos` itself is
  1.14e-05 and aatype 0/48) and pushes `multimotif_vp` to 0.00026 A, over its
  0.00020 A floor. All other configs and all other tests pass.

**n=160 structural gate: PASSES, both alone and combined** (job 907218, same
method and same seeds as section 1, three fresh arms off a frozen tree):

|arm|n|CA-CA (A)|Rg (A)|helix|strand|coil|clash|diverged|
|---|---|---|---|---|---|---|---|---|
|base|160|3.7635+-0.0131|12.87+-1.55|0.603|0.317|0.080|0.10|**0/160**|
|O8|160|3.7623+-0.0137|12.80+-1.23|0.573|0.337|0.090|0.17|**0/160**|
|**O2+O6+O8**|160|3.7625+-0.0127|12.80+-1.32|0.572|0.344|0.083|0.16|**0/160**|

Welch: every metric p >= 0.078 (O8) / p >= 0.135 (all). KS: every metric
p >= 0.33 / p >= 0.40. Fisher on diverged: p=1.0 both. MDE 0.097 helix /
0.44 A Rg against largest observed 0.031 / 0.07. One Brown-Forsythe flag, O8's
coil spread p=0.0254, which does **not** survive Bonferroni (alpha=0.0042) and
is p=0.31 in the combined arm on the same metric - noise. FIREDCHECK 0/160
identical PDBs in both arms, so both flags ran.

Compiled-vs-eager: 1.54x / 1.31x the floor at L=64 / L=100, aatype 0 differ.

**PORTABLE.** Plain PyTorch in `geometry/{frames,rigid,features}.py`; inductor
picks the block sizes. If anything it is *more* portable than the status quo,
because it stops depending on a vendor BLAS heuristic for a 3x3 problem. It
should pay MORE on a consumer card, where a kernel launch is relatively dearer
and cuBLAS's small-GEMM path is weaker - not measured (torch 2.5.1+cu121 cannot
target sm_120 at all).

## 5. "Batch across the 6 blocks where it is NOT a GEMM" - there is nothing left to batch

Read the loop (`DiffusionBlock.forward`): `pair_1`, `attn_1`, `update` and
`update_positions` all consume the **current** `pos`/`features`, so block i+1
cannot be evaluated until block i has finished. The only loop-invariant work in
a block is `pair_2` (it reads `prev_pos` and the hoisted `cond_neighbours`) and
`attn_2`'s frames of `prev_pos/sigma`.

And for those, **batching is strictly worse than what already shipped**: the
non-GEMM half of the six `pair_2` calls computes *identical values* in all six
blocks, so O1's `fx_graph_cse` collapses them 6->1 outright, rather than to one
batched kernel over a 6-tiled axis. The measured graph confirms it - the
`aten.stack`/`frames.py` kernels are down to 8 launches / 12.1 us, and the
`prev_pos/sigma` frames appear once.

What is left of the 6-fold structure is the per-block **weights**, which is a
GEMM question (P7, disabled) and N027's territory. **So: no, there is no
Triton-side equivalent of P7, because P7's Triton-side equivalent is CSE and it
is already landed.** Nothing here goes near the P7 code path.

## 6. Negatives, honestly

* **The GPU/DETSEG bitwise control has a non-zero self-floor** and cannot, on
  its own, make a bitwise claim. Two processes with the IDENTICAL configuration
  (`baserep` vs `base`, DETSEG=1, both compiled cold) differ by pos 6.15e-05 at
  L=64 and 8.96e-05 at L=100. Cause not diagnosed - cross-process cuBLAS
  heuristic selection is the suspect; `opt.DET_SEGMENT` is demonstrably read
  inside the traced region, so it is not the atomics. **Read against that floor
  the result is still decisive for O6** (job 907203):

  | | L=64 pos | L=100 pos | aatype |
  |---|---|---|---|
  | `baserep` (the self-floor) | 6.1512e-05 | 8.9645e-05 | 0 |
  | **O6** | **6.1512e-05** (= the floor, bit-for-bit, on every tensor) | **0.0000e+00 (BIT-IDENTICAL)** | 0 |
  | O8 | 1.0920e-04 (1.8x) | 1.8358e-04 (2.0x) | 0 |
  | O6+O8 | 1.9574e-04 (3.2x) | 1.8358e-04 (2.0x) | 0 |

  At L=100 O6 is bit-identical to base on the GPU while a RERUN of base is not,
  which is a stronger statement than `aggressive_fusion` ever got. The
  bit-exactness claim itself rests on the CPU compiled control (exactly 0.0,
  eleven outputs, L=64 and L=100, two independent jobs), which is what O1
  shipped on.
* **N017's O3 (the `pair_vector_features` cats) is still not reducible, and now
  it is costed**: `features.py:214` is 12 launches / 88.2 us and
  `features.py:210` is 12 / 49.2 us, together **137 us = 13 % of Triton time**,
  pure `aten.cat`. O6 halves the work feeding them but does not remove them.
  Splitting `linear_pairvec` into two projections to avoid the final cat trades
  one ~7.3 us cat for one 5.98 us GEMM launch plus a reassociation: costed at
  roughly +1.3 us, i.e. a loss. Writing the concatenation as one masked
  pointwise expression needs `local * (1/norm)` where the source has
  `local / norm`, which is not bit-exact. Left open, with the number attached.
* **The stable argsort** (19 launches / 61.9 us) is unchanged and remains a dead
  end: `topk` was +5.6 % (N011) and `combo_kernels` crashes on this node.
* `basic.py:96` (the HkLinear bias add, 48 launches / 101.4 us) and
  `basic.py:147` (LayerNorm, 33 / 84.5 us) are the two other big lines and are
  **N027's** file, not touched here.

## 7. What to ship

* **`exact` (and every preset): O6, `SALAD_OPT_PAIRFEAT=1` by default. -5.0 %,
  bit-exact, full parity suite `SUITE_EXIT=0`, 0/160 diverged.**
* **`fast`: O6 + O2 + O8 = -24.5 %**, subject to the pooled n=160 statistics for
  the `all` arm (job 907218) reading like O2's and O6's did.
* `SALAD_OPT_IPAMULSUM` and `SALAD_OPT_FRAMEMULSUM` default False and are
  opt-in; neither may ever enter `exact` (both fail a parity test).

## Reproduce

`x_fusion.py` + `x_parse_fusion.py` + `x_attrib.py` (the section-2 census),
`x_bench.py` (timings), `x_cpu_bits.py`/`x_cmp_bits.py` (the bitwise control),
`x_gpubits.py` (the GPU/DETSEG attempt), `x_ce.py` (compiled vs eager),
`stats.py` (N015's, verbatim), `apply_o8.py` (the O8 patch).
sbatch: `fusion` `bench` `bench2` `control` `control2` `quality` `quality2`
`gates` `ce` `gpubits` `o8diag`. Jobs: 907103 907111 907123 907143 907158
907159 907161 907165 907203 907206 907218.

## 8. LANDING (re-baselined on 40a85be, P7 re-enabled) - O6 as planned above was WRONG

Picked up after the previous agent died mid-task. Re-ran everything against
current HEAD (40a85be: P7 back on via `cond_prologue`'s fusion barrier, N026).
**O6 as section 3 shipped it is NOT bit-exact, and the "compiled `forward_core`
CPU control, max\|d\|=0.0" evidence above is real but insufficient** - the
exact defect **N029 independently found from a completely different angle
(§5.2-5.3 of `N029-l2-traffic/RESULTS.md`, its "O1")**: the two functions
contract at two different widths (`direction_features` is `bmm(100,3,3) x
(100,3,K*A)`, `pair_vector_features` is `bmm(100,3,3) x (100,3,K*2A)`), and
whichever call is rewritten to *reuse* the other's width picks a DIFFERENT
BLAS kernel for that width, DATA-DEPENDENTLY. N029 measured this at 1.63e-04 A
on `multimotif_vp`, above that config's own parity floor. Independently, on
CPU eager (the sharpest instrument - `test_sampling_parity.py`, chaotic
iterated map), the SAME defect surfaces as `final_aatype` going from 1/64
(pristine) to 7/64 (worst case, ratio 38.7x, hard FAIL against the test's own
<=3 threshold) - reproduced deterministically at 48 threads, isolated via a
two-process, block-by-block hook on `DiffusionPairFeatures.forward` with
IDENTICAL inputs: block 0 exact, block 1 exact, **block 2 diverges by 1.8e-7
with bit-identical `pos` entering it** - and a FRESH process replaying the
exact same captured tensors reproduces the reference exactly, proving it is
call-CONTEXT-dependent kernel selection, not a value or a logic bug. Every
reshuffle tried (batch self+neigh together and slice; transform separately at
each function's own width and concatenate the RESULTS) reproduced the same
class of divergence the moment the (2A)-wide contraction was touched at all.
**N029's §5.3 ("there is no bit-exact version of this optimisation") is
correct for that mechanism** - sharing or moving the WIDE contraction.

**What ships instead: share only what has no rounding.** `aa_frames_t(pos)`
(a per-residue frame, independent of K) and `neigh_pos = pos[neighbours]` (a
pure index gather) are reused between the two functions; each function's own
`apply_inverse_t`/`vec_normalized` call runs UNCHANGED, at its ORIGINAL width
and call count - `direction_features` still contracts at width `A`,
`pair_vector_features` still at width `2A`, exactly as shipped today. This
sidesteps N029's mechanism entirely (no width is ever moved or shared) and is
**verified bit-exact three ways**: function-level bit-exact (as before,
insufficient alone); whole compiled `forward_core`, CPU, all eleven outputs,
L=64 and L=100, max\|d\|=0.0 (job 907814); AND, the check the naive version
failed, eager `test_sampling_parity.py` on CPU produces **bit-identical
trajectories** at `SALAD_OPT_PAIRFEAT=0` and `=1` (1/64, ratio 7.3x, both -
literally the same numbers to every printed digit, not just both passing) -
confirmed in two independent jobs (907814, 907816).

**Cost of being honest about the mechanism: most of the claimed win is gone.**
The "twelve wide `apply_inverse_t` bmm" launches N029 identifies as 96% of
O1/O6's -7.3%/-5.0% are exactly the launches this safe version does NOT
remove (both functions still call `apply_inverse_t` at their own width). What
is actually saved - one `aa_frames_t` call and one neighbour gather per pair
block - is real but cheap (O(N), not O(N*K), and a pure memory op) and is
**not resolvable from noise at L=100** (job 907815, `exact` preset, 2 reps:
3.076/3.127 ms vs the pre-O6 HEAD baseline of 3.074 ms in the task brief - the
same number within run-to-run variance). It ships anyway because it is
genuinely free (bit-exact, harmless) and matches `aggressive_fusion`'s
"default everywhere, veto-only" class - not because it moves the timing.

**Gates, this landing (jobs 907814 CPU, 907815 GPU, both against current
HEAD with the corrected code):** compiled CPU bitwise control 0.0/0.0 (L=64,
L=100); eager `test_sampling_parity` bit-identical both flag states; full
parity suite (4 configs + geometry/stack/encoder/sampling) `EXIT 0`
everywhere on both CPU and GPU; `test_design_quality.py --n 160 --preset
fast` 0/160 diverged; `--n 64 --preset exact` 0/64 diverged.

**Also fixed while landing:** the `IPA_MULSUM`/`FRAME_MULSUM` env-var
combination was written as `preset OR env` (a measurement-phase convention,
appropriate for A/B-ing a candidate against any preset without editing the
preset dict) but never flipped to the project's shipping convention -
`preset AND env`, env default True - before landing, which means
`SALAD_OPT_FRAMEMULSUM=0`/`SALAD_OPT_IPAMULSUM=0` could not veto either flag
once a preset turned it on. Fixed to match `BATCH_COND`'s pattern exactly
(`_flag(name)` default True, `opt.X = bool(p.get(x)) and _ENV_X`).
`PAIR_FEAT_MERGE` is now also a proper preset key (`True` in `exact`/`fast`/
`deterministic`, alongside `aggressive_fusion`) rather than a bare
module-level default with no preset entry, for the same operational-veto
reason. Caught in review before any commit landed; HEAD was clean throughout.

O8 and O2's own findings (sections 4 and "O2's verdict" above) are
**unaffected** by any of this - neither shares a contraction across two
different call sites, each is a single-call reassociation, and their CPU
bitwise controls (NOT bit-exact, as documented) and n=160 structural gates
stand as measured.
