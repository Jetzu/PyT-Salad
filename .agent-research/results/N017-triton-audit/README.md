# N017 - Triton kernel audit: the 524 unaudited inductor kernels

Target: the 524 Triton kernels in one captured replay of the compiled denoiser
core. All previous optimisation effort (P1-P7) went at the 321 cuBLAS launches.

Baselines, both at L=100, `default_vp`, preset `exact`, lean step
(`need_aa=False`, `pair_cond=False`), H100:

| tree | job | kernels | triton | gemm | memcpy | GPU busy |
|---|---|---|---|---|---|---|
| committed HEAD a8533a7 (= the N012/job-903203 baseline) | 903957 / 903972 | 921 | 524 | 359 | 36 | 3432 us |
| live working tree (P7 landed, uncommitted) | 903993 | see `live/` | | | | |

Files:
* `attribution_table.txt` - the attribution, by stage and by innermost source line
* `kernel_rows.json`, `kernel_attr_counts.json` - the raw per-kernel rows
* `profile_lean.json` - the profiler census + the launch-floor probe
* `inductor/` - inductor's `output_code.py` + `fx_graph_readable.py` for HEAD
* `live/` - the same for the live snapshot
* `x_attrib.py`, `x_parse.py`, `x_bench.py`, `x_cpu_bits.py` - the harness
* `live_snapshot_md5.txt` - exactly which file versions were measured

## 1. What the 524 kernels are (HEAD a8533a7, job 903972)

524 Triton launches of **163 distinct kernels**, 1244 us of the 3432 us the
captured lean step spends on the GPU (36 %). The other 64 % is 359 cuBLAS
launches (2146 us) and 36 memcpys (44 us).

**The marginal price of one kernel inside a captured graph is 1.06 us**
(no-op Triton kernel replayed 0/50/100/200/400 times in one capture:
12.3 / 61.0 / 116.0 / 224.6 / 437.9 us; slope 1.064 us, same job). So the 524
Triton kernels carry **~557 us of pure "this kernel exists" cost** - 45 % of
all Triton time and 16 % of the whole step - before any of them does work.
The average Triton kernel costs 2.38 us, i.e. barely twice the floor.

The same probe prices a cuBLAS launch: the 359 GEMM launches average 5.98 us
each for problems whose arithmetic is well under 1 us. **That is the single
most important number in this audit**: swapping a Triton kernel for a GEMM
launch loses 5x, so no "fuse this into the matmul by splitting the GEMM"
transformation can pay unless it removes a GEMM as well.

### Fusion state
From `ir_pre_fusion.txt` / `ir_post_fusion.txt`: 1028 pointwise/reduction
scheduler nodes go in, and come out as **193 fused groups + 331 single
unfused nodes = 524 kernels**. So **63 % of the Triton launches carry exactly
one ATen op**, and the mean fused group holds 3.6.

### Attribution table (HEAD a8533a7, lean captured step, L=100)

By stage (a kernel is charged to the call site of its first fused node;
`blocks.py:158-224` is `DiffusionPairFeatures.forward`, reached from BOTH
`pair_1` and `pair_2` on this tree - P7 was not yet committed here):

| stage | launches | us | % of triton | distinct kernels |
|---|---|---|---|---|
| pair features (`DiffusionPairFeatures`) | 213 | 536.5 | 43.1 % | 66 |
| IPA attention | 134 | 385.8 | 31.0 % | 35 |
| global update (`Update`) | 48 | 84.0 | 6.8 % | 5 |
| block glue / neighbour fn | 48 | 72.7 | 5.8 % | 24 |
| neighbour stable argsort (`aten.sort`) | 19 | 62.3 | 5.0 % | 1 |
| position update | 33 | 56.4 | 4.5 % | 16 |
| `forward_core` prologue | 28 | 43.6 | 3.5 % | 15 |

By innermost source line (top items):

| launches | us | source |
|---|---|---|
| 45 | 129.0 | `basic.py:117` LayerNorm normalise |
| 12 | 87.4 | `features.py:214` `pair_vector_features` final `cat` |
| 12 | 80.6 | `attention.py:109` `einsum("ihc,ijhc->ijh", q, k[nb])` |
| 12 | 71.3 | `attention.py:108` `dist` reduction over the query points |
| 19 | 62.3 | the stable argsorts |
| 30 | 56.5 | `blocks.py:258` `index_mean` inside `Update` |
| 42 | 54.2 | `frames.py:70` `aa_frames_t` |
| 12 | 51.1 | `features.py:210` `pair_vector_features` `cat` of self+neighbour atoms |
| 31 | 44.1 | `rigid.py:86` `rot_from_two_vectors`'s `stack` |
| 12 | 42.5 | `features.py:56` the distance-RBF expansion |
| 25 | 32.0 | `rigid.py:56` `vec_cross` |

**Kernels whose innermost frame is `rigid.py` or `frames.py` - i.e. pure 3x3
frame algebra on a (100, 3) tensor - are 161 launches and 227 us, 18 % of all
Triton time.**

## 2. The same audit on the LIVE tree (P7 landed, uncommitted) - job 903993

P7 (`StackedCondPair`) has landed in the working tree but is not committed, so
the shipping configuration is not the one N012 measured. Re-run there:

| | HEAD a8533a7 | live tree (P7) |
|---|---|---|
| kernels | 921 | **727** |
| triton launches / distinct | 524 / 163 | **381 / 198** |
| cuBLAS (all `extern_kernels.bmm`) | 359 (303 bmm calls) | **308 (258 bmm calls)** |
| memcpy | 36 | 36 |
| GPU busy | 3432 us | **3020 us** |
| lean step (end to end) | - | **3.154 ms** |

Triton by stage on the live tree (381 launches, 1079 us):

| stage | launches | us | % triton |
|---|---|---|---|
| IPA attention | 121 | 351.9 | 32.6 % |
| pair features (`pair_1`) | 89 | 237.0 | 22.0 % |
| P7 cond-pair prologue | 24 | 196.7 | 18.2 % |
| block glue / neighbour fn | 51 | 85.5 | 7.9 % |
| global update (`Update`) | 30 | 67.0 | 6.2 % |
| neighbour stable argsort | 19 | 61.6 | 5.7 % |
| position update | 27 | 43.3 | 4.0 % |
| `forward_core` prologue | 16 | 23.0 | 2.1 % |

Note what P7 did to its own stage: 24 launches for 196.7 us, i.e. 8.2 us each
against a 1.10 us floor. It converted a launch-bound stage into a
**memory-bound** one (one `bmm` kernel at 44.7 us, one `cat` at 31.6 us, one
RBF expansion at 21.0 us, each a single launch over the 6x-tiled residue
axis). That is the right trade on this card and a WORSE one on a consumer
card - see section 6.

## 3. The redundancy that dominates: 61 Gram-Schmidt frames per forward

`aa_frames_t(pos)` -> `rot_from_two_vectors` -> normalise, dot, normalise,
cross, `stack`, then `apply_inverse_t`. On a `(100, 3)` tensor this is a few
thousand flops and **4-5 kernel launches**.

`fx_graph_readable.py` contains **61 separate `aa_frames_t` invocations** on
HEAD (46 on the live tree, P7 having merged three of them). Inductor does NOT
common-subexpression them: `joint_graph_passes` runs `remove_noop_ops`, the
pattern matcher and constant folding, but no CSE. The duplicates are:

| call site | per forward | distinct values |
|---|---|---|
| `DiffusionPairFeatures.forward` via `direction_features`, `position_rotation_features`, `pair_vector_features` | 3 x 6 blocks = 18 | 6 |
| `Update.forward` | 6 | (same 6) |
| `UpdatePositions.forward` | 6 | (same 6) |
| `SparseStructureAttention` on `pos/sigma` (attn_1) | 6 | 6 |
| `SparseStructureAttention` on `prev_pos/sigma` (attn_2) | 6 | **1** |
| `StackedCondPair` on the tiled `prev_pos` | 3 | 1 |
| `forward_core` prologue | 1 | 1 |

so ~31 of the 61 are recomputations of a value already in a register file
somewhere else in the same graph. Kernels whose innermost frame is
`rigid.py`/`frames.py` are 161 launches / 227 us on HEAD (18 % of Triton time)
and 102 launches / 151 us on the live tree (14 %).

## 4. Opportunities, ranked, with PORTABILITY marked

The project must run on consumer Ampere/Ada (sm_80/86/89) and Blackwell
(sm_120), so every item is marked PORTABLE (no arch-specific code, no tuned
constant, inductor generates and autotunes the kernel) or H100-DEPENDENT.

| # | opportunity | launches it removes | class | portability |
|---|---|---|---|---|
| O1 | **graph-level CSE of the 61 duplicate frame computations** | -43 triton, -7 bmm | bit-exact | **PORTABLE** - an FX-graph pass, runs before codegen, no Triton, no constants, no arch |
| O2 | IPA's four contractions: `einsum` -> mul+sum, so the gathered neighbour tensors are read in place instead of being `clone`d into bmm layout | -4 bmm and ~-6 `clone` per attention branch | reassociation | **PORTABLE** - plain PyTorch source, inductor generates and autotunes |
| O3 | `pair_vector_features`' two `cat`s (the `(N,K,240)` one is 6.1 MB written then re-read by a GEMM); the self half is K-independent and is materialised K times | ~-3 triton per pair block | reassociation | PORTABLE, but see below - it converts triton kernels into extra GEMM launches |
| O4 | `index_mean`'s `(N, F)` `ones_like` denominator -> `(N, 1)` | measured: **+9**, see section 5 | bit-exact | PORTABLE |
| O5 | fuse the distance-RBF expansion into the following GEMM (hand-written matmul with an RBF prologue) | -1 triton, -0 bmm per pair block | reassociation | **H100-DEPENDENT as a hand-written kernel** - would need its own autotune space per arch, and it has to beat cuBLAS on a `(6400,400)@(400,64)`. Ruled out under the portability constraint. |

Two structural numbers decide the ranking:

* a Triton kernel inside a captured graph costs **1.06 us** just to exist
  (measured, section 1);
* a cuBLAS launch costs **5.98 us** on average for problems whose arithmetic is
  under 1 us.

So **any transformation that trades a Triton kernel for a GEMM launch loses
about 5x**. That kills the whole "split the GEMM so the elementwise work fuses
into it" family, which is why O3 is ranked below O2 despite being larger in
bytes, and why O5 is not worth a hand-written kernel even before the
portability constraint rules it out.

`combo_kernels` (inductor's horizontal fusion, the obvious lever against 331
single-op kernels) is already ruled out: it **crashes** on the stable-argsort
node (`ops.sort is only supported in persistent reductions`), and the variant
that avoids the crash by switching to `topk` measured 2.978 vs 3.007 ms, i.e.
nothing (N011, job 903189). Not re-tested here.

## 5. Prototypes: what was measured

All on the live snapshot, L=100, `default_vp`, preset `exact`, one arm per
process (job 903992, 2 reps each). "lean" is the whole-forward lean step, not
an isolated kernel.

| arm | lean ms | vs base | kernels | triton | gemm | GPU busy |
|---|---|---|---|---|---|---|
| base | 3.153 / 3.155 | - | 727 | 381 | 308 | 3006 us |
| **O1 `fx_graph_cse`** | **3.072 / 3.064** | **-2.7 %** | **677** | **338** | **301** | **2922 us** |
| O1 via `TORCHINDUCTOR_FREEZING=1` | 3.074 / 3.050 | -2.9 % | 677 | 338 | 301 | 2918 us | <- same speed but **NOT bit-exact**, see below |
| O4 `index_mean` (N,1) denominator | 3.156 / 3.158 | **+0.1 % (nothing)** | **736 (+9)** | 390 (+9) | 308 | 3010 us |
| O1 + O4 | 3.066 / 3.067 | -2.8 % | 678 | 339 | 301 | 2923 us |

### O1 - graph-level CSE: TAKEN. -2.7 %, bit-exact, PORTABLE.

One line, no source change, no Triton:

```python
import torch._inductor.config as ind
from torch._functorch.compile_utils import fx_graph_cse
ind.joint_custom_post_pass = (
    lambda g: setattr(g.owning_module, "graph", fx_graph_cse(g)))
```

It removes **43 Triton launches and 7 bmm launches**, which is exactly the
31 duplicate frame computations predicted in section 3 (the six blocks'
`pair_1`/`update`/`update_positions` collapsing 5->1, `attn_2`'s six
`prev_pos/sigma` frames collapsing 6->1, and the prologue's 3->1).

**Control: bit-identical.** CPU is deterministic by construction, so this is a
bitwise answer rather than a tolerance. `x_cpu_bits.py`, the whole compiled
`forward_core`, at **L=64 and L=100** (job 904001) and again at L=100 in a
separate job (904080): `bit_identical=True`, `max|d| = 0.000e+00` on every one
of the eleven outputs. That is the correct control for a compile-time graph
transformation - the four checkpoint parity tests run the EAGER path and cannot
see this change at all.

**Not an L=100 artefact** (job 904079, one length per process):

| L | base lean | +CSE | delta | kernels |
|---|---|---|---|---|
| 50 | 2.573 ms | 2.498 ms | **-2.9 %** | 745 -> 695 |
| 100 | 3.154 ms | 3.068 ms | **-2.7 %** | 727 -> 677 |
| 200 | 4.459 ms | 4.346 ms | **-2.5 %** | 715 -> 665 |

The kernel count falls by exactly 50 at every length, as it must - the
duplication is a property of the graph, not of the shape.

### !! `TORCHINDUCTOR_FREEZING=1` IS NOT BIT-EXACT - do not use it for this !!

`freezing` runs `fx_graph_cse` internally and measured the same speed (3.074 /
3.050 ms, 677 kernels), so it looks like a drop-in, documented alternative.
**It is not.** Job 904080, same CPU bitwise control at L=100:

```
CPUBITS2 L=100 cse  bit_identical=True
CPUBITS2 L=100 frz  bit_identical=False
   pos          max|d|=8.821e-06      local       max|d|=5.493e-04
   trajectory   max|d|=8.821e-06      raw_angles  max|d|=1.526e-03
   angles       max|d|=2.296e-05      angle_pos   max|d|=8.047e-06
```

Freezing does more than CSE - it folds the parameters into constants and runs
its own layout/binary-folding passes, and those change the arithmetic. The
deviation is small (8.8e-06 A on `pos`, under the 2e-05 A parity floor, so it
would very likely *pass* `test_model_parity`) which is exactly what makes it
dangerous: it would have shipped inside `exact` looking clean. **The narrow
`joint_custom_post_pass` form is strictly better - identical speed, identical
kernel count, and actually bit-identical.** This is the one place in this item
where the documented knob is the wrong answer and the private one is right.

`freezing` also bakes the weights in, so re-loading a checkpoint into an
already-compiled model would need a recompile. A second reason to prefer the
narrow form.

**Portability: PORTABLE.** It is an FX-graph rewrite that runs before codegen.
No Triton is written, no BLOCK/num_warps/num_stages constant is chosen, no
shared-memory or SM-count assumption is made, and the kernels inductor emits
afterwards are still inductor's own - autotuned per device as before. Equally
valid on sm_80/86/89 and sm_120.

**Version exposure.** `joint_custom_post_pass` and
`torch._functorch.compile_utils.fx_graph_cse` are both private paths that could
move between torch releases. Neither depends on Triton's codegen, so a Triton
version bump cannot break it; a torch bump can, and the failure mode is an
`AttributeError` at import, i.e. loud. Guard the import and fall back to the
plain path. The thing to re-check after a torch upgrade is whether the newer
inductor has started CSE-ing inference graphs by itself, in which case O1
becomes a no-op and should be deleted rather than carried.

**The durable form is to do this CSE in the SOURCE** (thread `rot, trans`
through `DiffusionPairFeatures`/`Update`/`UpdatePositions` instead of calling
`aa_frames_t` five times per block, and hoist the `prev_pos/sigma` frames out
of the six-block loop). That needs no torch-private API at all. The -2.7 %
above is its ceiling; the source version is NOT separately measured here and no
number should be quoted for it until it is.

### O4 - `index_mean`'s denominator: REJECTED, measured.

`index_mean(data, index, mask)` builds `weight = torch.ones_like(data)`, an
`(N, F)` block of ones, and scatter-adds it to get a per-segment COUNT - the
same number in every one of F columns. Replacing it with an `(N, 1)` column is
algebraically identical and bit-identical in practice (a sum of 1.0s is exact
in fp32, and `data * 1.0 == data`).

It is bit-exact: `bit_identical=True, max|d|=0.000e+00` on CPU at both lengths
(job 904001), and `test_sampling_parity` prints a trajectory identical to
baseline digit for digit (job 904010).

**And it makes things worse.** 736 kernels against 727, +9 Triton launches,
+0.003 ms. The `(N, F)` `ones_like` was never costing anything: inductor was
folding it into the `(N, F)` pipeline it already had. The `(N, 1)` version is a
differently-shaped buffer that can no longer fuse with its `(N, F)` neighbours,
so it becomes its own kernels. This is the audit's cleanest reminder that
**"obviously wasted work" and "a kernel" are different currencies**, and only
the second one is being paid for here.

### O2 - the IPA contractions: DISQUALIFIED from `exact` on the parity gate.

`SparseInvariantPointAttention.forward` contracts four times with
`torch.einsum`. ATen lowers every one of them to `bmm`, and because the
operands are gathered neighbour tensors in the wrong layout, inductor has to
materialise a transposed copy first. `triton_poi_fused_clone_18` is the single
most expensive Triton kernel in the live tree: **5 launches, 11.19 us each**,
purely to lay out `k[neighbours]` (a 3.3 MB tensor) for a `bmm`. The same
pattern repeats for `v[neighbours]` and `v_g[neighbours]`.

Writing them as explicit mul+sum lets inductor read the gather in place and
emit one fused reduction, removing 4 `bmm` launches and ~6 `clone` kernels per
attention branch:

```python
dot        = math.sqrt(1.0 / s) * (q[:, None] * k[neighbours]).sum(-1)
out_pair   = (attn[..., None] * pair[:, :, None, :]).sum(1)
out_scalar = (attn[..., None] * v[neighbours]).sum(1)
out_point  = (attn[..., None, None] * v_g[neighbours]).sum(1)
```

Equivalent to fp64 rounding (`max|d| = 1.78e-14` against the einsum form on a
random fp64 instance), i.e. a pure reassociation of the reduction order.

**It fails the gate.** `tests/test_sampling_parity.py`, job 904010:
`final_aa rel = 7.89e-02`, **7 of 64 residues take a different amino acid**,
final structure mean 0.265 A / max 0.623 A against a 0.0065 A self-divergence
floor (40.8x). The four checkpoint `test_model_parity` configs, geometry 38/38,
stack and encoder all still PASS - a single forward is fine; it is the 81-step
rollout that amplifies. This is the same failure signature and the same
magnitude as `SALAD_OPT_DETSEG=1` (N015: 7/64 residues, 30.6x), which the
project already decided is a separate preset and not `exact`.

So it can only ever be a `fast`-preset lever, and only if it pays. It does pay - see below.

**Portability: PORTABLE.** It is four lines of ordinary PyTorch. No hand-written
Triton, no tuned constants - inductor generates the reduction and picks its own
block sizes per device, which is exactly the "let inductor autotune" route the
constraint asks for.

### O5 - a hand-written fused RBF-matmul: NOT ATTEMPTED, ruled out.

`distance_features` materialises an `(N, K, 400)` RBF expansion (10.2 MB) that
the next GEMM immediately reads back. Fusing the expansion into a matmul
prologue would save ~20 MB of traffic per pair block. That needs a hand-written
Triton matmul, and under the portability constraint it is not worth it: it
would have to carry its own `triton.autotune` space per architecture, it must
beat cuBLAS on a `(6400,400)@(400,64)` on every target card, and a hand-written
matmul is the single most version-exposed thing one can write against Triton
(block/stage/warp semantics and the `tl.dot` lowering all move between Triton
releases). **H100-DEPENDENT as written; ruled out.** Marked here so the next
person does not rediscover it as free money.

**No hand-written Triton kernel is proposed by this item.** The only Triton I
wrote is the launch-floor probe in `x_attrib.py`, which is a measuring
instrument, not shipped code (and it uses only `tl.load`/`tl.store`/
`tl.arange`/`program_id`, so it would run anywhere, but it never runs in the
model).

## 6. Consumer-GPU / Blackwell portability of the AUDIT ITSELF

Two numbers in this report are properties of this H100 and must not be carried
to another card:

* the **1.06 us** per-kernel floor in a captured graph;
* the **5.98 us** average cuBLAS launch.

What is NOT card-specific is the shape of the problem. The step does 0.21 ms of
arithmetic and 0.37 ms of memory traffic (N007) inside a 3.0-3.4 ms wall, so it
is launch-bound by roughly 10x, and 63 % of its Triton launches carry a single
ATen op. Kernel COUNT is the currency on any device.

The balance does shift, and it shifts **towards** the work this item points at.
A consumer card has far less memory bandwidth relative to its fp32 rate than an
H100 (e.g. ~1 TB/s vs 3.35 TB/s, against broadly comparable non-tensor fp32
throughput), so the memory-bound kernels get relatively MORE expensive there:

* the `pair_vector_features` `cat` (6.1 MB written and re-read, 8.18 us here),
* the distance-RBF expansion (10.2 MB, 3.80 us here),
* **P7's prologue** - 196.7 us in 24 launches on the live tree, i.e. already
  memory-bound rather than launch-bound (one `bmm` kernel at 44.7 us).

P7 traded launches for bandwidth on the 6x-tiled residue axis. That is the
right trade on an H100 and a **worse** trade on a consumer card, and someone
should re-measure P7's -9.2 % on the real target before assuming it holds.
Flagging it because it is the largest existing H100-shaped assumption in the
fast path, and it is not mine to change.

O1 is indifferent to all of this: it deletes duplicated work, so it wins on
whatever device the work was being duplicated on.

**torch 2.5.1+cu121 cannot target sm_120 at all** (Blackwell needs CUDA 12.8+
and torch >= 2.7 with a newer Triton), so nothing here is tested on Blackwell.
That matters far less than it would for a hand-written kernel, because O1 is an
FX-graph pass and O2 is plain PyTorch - both are re-lowered by whatever
inductor/Triton the target ships. The one thing to re-check after a torch
upgrade is whether the newer inductor has started CSE-ing inference graphs by
itself, in which case O1 becomes a no-op and should be dropped rather than
carried.

### O2 measured: it pays, and it composes with O1 - but only for a deviating preset

Job 904079, live snapshot, L=100, 2 reps:

| arm | lean ms | vs base | kernels | triton | gemm |
|---|---|---|---|---|---|
| base | 3.178 / 3.150 | - | 727 | 381 (1076 us) | 308 (1901 us) |
| O2 (IPA rewrite) | 3.066 / 3.063 | **-3.1 %** | 667 | 405 (1226 us) | **260** (1707 us) |
| **O1 + O2** | **2.976 / 2.979** | **-5.9 %** | **618** | 363 (1162 us) | **253** (1689 us) |

This is the launch-economics argument running in the right direction for once:
O2 **adds 24 Triton launches and removes 48 cuBLAS launches**, and 48 x 5.98 us
beats 24 x ~1.06 us plus the extra reduction work by a wide margin. GEMM time
drops 1901 -> 1707 us; Triton time rises 1076 -> 1226 us; the step drops 0.10 ms.

O1 and O2 compose almost additively (-2.7 % and -3.1 % separately, -5.9 %
together), which they should: they touch disjoint kernels.

**But O2 changes the design.** `test_sampling_parity` fails at 7/64 residues
(job 904010). So the honest split is:

* **`exact` gets O1 only: -2.7 %** (bit-identical, no gate to argue about);
* **a deviating preset could take O1+O2: -5.9 %**, and would need the same
  n=160-per-arm structural validation N015 ran for `steps_scale=0.5` before
  anyone trusts it. I have NOT run that, so O2 is a measured candidate, not a
  recommendation.

## 7. What is left, and is it reducible

After O1 the lean step is **3.068 ms** (from 3.154), 677 kernels: 338 Triton
(1002 us), 301 cuBLAS (1877 us), 36 memcpy, 2922 us GPU busy against ~450 us of
wall that is not GPU-busy at all.

Where the remaining 338 Triton launches sit, and what each is worth:

| what | launches | us | reducible? |
|---|---|---|---|
| IPA attention | ~121 | ~352 | **yes, -3.1 %**, by O2 - but it costs the sampling-parity gate |
| pair features `pair_1` | ~89 | ~237 | partly. The two `cat`s in `pair_vector_features` are 90 us of pure data movement, and the self half of `both` is K-independent and materialised K times. Every way of removing them that I costed adds a GEMM launch, and a GEMM launch is 5.98 us against a 1.06 us Triton kernel, so it loses. Not reducible without a hand-written fused matmul, which the portability constraint rules out. |
| P7 cond-pair prologue | 24 | ~197 | not by me - it is another agent's live work, and it is already memory-bound rather than launch-bound (8.2 us/launch). Flagged in section 6 as the largest H100-shaped assumption in the fast path. |
| neighbour stable argsort | 19 | ~62 | no. `topk` was measured +5.6 % (N011) and `combo_kernels` crashes on this node. |
| global update, position update, prologue | ~85 | ~155 | ~30 of these are the surviving frame computations. 6 of them (`attn_1`'s `pos/sigma` frames) are mathematically equal to 6 others (a Gram-Schmidt frame is scale-invariant) but not bit-equal, so merging them is a numerics change worth maybe 8 kernels / 0.4 %. Not worth the gate. |

**The honest ceiling.** 301 cuBLAS launches at 5.98 us each is 1800 us, i.e.
**59 % of the remaining GPU-busy time is cuBLAS launch latency on GEMMs whose
arithmetic is under 1 us**. The Triton half, which this item was pointed at, is
338 launches x 1.06 us = 358 us of floor plus ~640 us of real work. So:

* the Triton side still has ~250-350 us of removable launch floor in it, but
  every remaining route to it either fails the parity gate (O2), trades into
  more expensive GEMM launches (O3), or needs a hand-written arch-specific
  kernel (O5);
* **the next real factor is not on the Triton side at all.** It is the 301
  batch-1 `bmm` calls. Every `HkLinear` is
  `torch.einsum("...k,ki->...i", x, w)`, which ATen lowers to `bmm` with
  batch=1 - there is not a single `mm` or `addmm` in the whole generated
  module. That forecloses inductor's own matmul templates (and its per-device
  autotuning, which is exactly what the portability constraint asks for) and
  its bias epilogue fusion. N012's standalone GEMM-batching test measured
  1.87-2.13x on this, and P7 is the first 48 launches of it. **That is where
  the remaining 2x lives**, and it is a bigger prize than anything left in the
  Triton half.

So: the 524 Triton kernels were worth auditing - they yielded a free,
bit-exact, fully portable 2.7 %, and a measured 5.9 % if a deviating preset is
acceptable - but they are **not** where this model's remaining time is. The
brief's premise that "the 524 Triton kernels are the LARGER half" is true by
count and false by cost: they are 36 % of GPU-busy time against cuBLAS's 62 %.

---
Inductor's artefacts under `inductor/` (HEAD a8533a7) and `live/inductor/`
(live snapshot) are gzipped; `zcat inductor/*/output_code.py.gz` is the
generated module, `zcat inductor/*/fx_graph_readable.py.gz` the post-dispatch
FX graph with `# File:` annotations, and the two `ir_*_fusion.txt.gz` are the
scheduler's node lists before and after fusion (the source of the 193 fused
groups + 331 single nodes = 524 kernels count).

Reproduce: `sbatch attrib.sbatch` (profile + debug dump), then
`python x_parse.py`; arms with `bench.sbatch` / `bench2.sbatch`; controls with
`cpubits.sbatch` / `cpubits2.sbatch` / `parity.sbatch`.
