# N016 — the per-step lower bound for `exact` and `fast`, derived and measured

All numbers: H100 (132 SM, sm_90, 60 MiB L2, 94 GiB, max SM clock 1980 MHz),
node h11-24, torch 2.5.1+cu121, `default_vp-200k`, **L=100**, one length per
process, detached worktree at HEAD `a8533a7`
(`/hnvme/workspace/v181bb10-salad/n016-floor`). Jobs 903966–904019. No
production code was changed.

## 0. The verdict in one table

|  | `exact` | `fast` |
|---|---|---|
| measured lean step (wall, production `FastDenoiser`) | **3.573 ms** | **3.007 ms** |
| — captured graph replay | 3.395 ms | 2.835 ms |
| — python wrapper | 0.178 ms | 0.172 ms |
| **LOWER BOUND, per step** (§6) | **0.258 ms** | **0.239 ms** |
| factor above the bound | **13.8×** | **12.6×** |
| lower bound, per 100-residue design | 0.104 s (401 steps) | 0.048 s (201 steps) |
| capacity floor of today's kernels (§7) | ~1.66 ms | ~1.39 ms |

**The per-step bound is within 7 % the same for both presets** (0.258 vs
0.239 ms at L=100) — and that is not an accident: the binding term is
*serialisation*, and every relaxation `fast` is allowed to make (reduced
precision, fewer neighbours, fused approximations) attacks *arithmetic* or
*memory traffic*, both of which are already two orders of magnitude below the
measured time. `fast`'s real win is per **design**, not per step: halving the
schedule halves the per-design bound. Per step it buys 0.56 ms of
implementation slack, not floor.

**The N009 "3.4–3.5 ms hard floor" is refuted a second time, and in the other
direction from N012.** N012 showed the claim was unproven. This item shows it
is *wrong by an order of magnitude*: the real bound is 0.26 ms, and even the
capacity floor of the current, unchanged kernel set — what these 946 kernels
cost when the GPU is not idle — is 1.66 ms. Nothing about
this model's mathematics or this GPU's physics prevents a sub-millisecond step.

---

## 1. Method: what a bound is, and which one each number is

Three classes, and every number below is labelled with which it belongs to.

* **B0 — problem bound.** True for *any* program on this GPU that computes this
  function. Only arithmetic and memory traffic.
* **B1 — kernel-granularity bound.** True for any implementation whose global
  synchronisation is a kernel boundary or a grid barrier — i.e. everything
  writable in PyTorch, Triton, CUDA C or JAX. Adds the dataflow serialisation
  term.
* **B2 — this-implementation floor.** Holds the current kernel decomposition
  fixed. **Not a bound on the problem**; it is what N009 reported as if it were
  one, which is exactly the error N012 caught.

The bound is
`T >= max( FLOPs / peak , bytes / BW , D * c_sync )`
with every term on the right measured on this node, not estimated. It is a
`max`, not a sum: the three resources can overlap in time, so only the largest
is guaranteed.

---

### 1.1 What the parity constraint does and does not cost

The brief asks for the irreducible kernel count *given the architecture and the
parity constraint*. The answer is that **parity costs nothing in the bound**:

* `exact`'s constraint is a **numerical** one — fp32 accumulation, no
  reassociation of a reduction whose order the JAX reference fixes. It forbids
  TF32 and it forbids turning six `mm`s into a `bmm` that picks a different
  split-K. It does **not** forbid fusion, batching that preserves order, kernel
  replacement that preserves order (§9's Triton templates are `max|d| = 0.0`
  precisely for this reason), or scheduling.
* The barrier count `D` is a property of the **dataflow**, which is the same
  graph under either preset. So parity changes neither `D` nor the traffic.
* Where parity does bite is the *arithmetic rate*: fp32 FMA at 66.91 TFLOP/s
  instead of TF32 tensor core. That is 0.150 vs 0.020 ms at L=100 — 0.13 ms,
  and not binding (§2.4). At L=400 it is 0.599 vs 0.081 ms, and it *is*
  binding, which is the whole of the presets' floor difference (§9c).

And on the **total** kernel count, as distinct from the depth: it is bounded
below by `D = 205` (barriers on the chain cannot be merged) and above by the
779 barrier nodes in the graph (merging nothing). Today's forward launches
**946**, i.e. 1.2× the no-merging count — so there is room for roughly a
**4.6× reduction** in launches before the depth makes it impossible.

---

## 2. The three terms, measured

### 2.1 Arithmetic — hard, and small (B0)

`FlopCounterMode` over the shipping lean `forward_core` (`x_census.py` §F,
job 903980):

```
F lean: matmul_FLOPs = 10.0140 GFLOP   (full/reporting step: 11.6033 GFLOP)
```

fp32 FMA peak on this node is `132 SM x 128 cores x 2 x 1.980 GHz` =
**66.91 TFLOP/s** (clock read from `nvidia-smi`, job 903977):

| | GFLOP | peak | floor |
|---|---|---|---|
| `exact` (fp32, `matmul_precision=highest`) | 10.014 | 66.91 TFLOP/s | **0.150 ms** |
| `fast` (TF32 tensor core) | 10.014 | 494.7 TFLOP/s (vendor dense) | **0.020 ms** |

Non-matmul arithmetic is not counted; it is elementwise over ~1.3 G output
elements at a few FLOP each, which at peak is well under 0.05 ms and is in any
case bounded by the memory term below. **Hard.** Nothing can beat peak.

### 2.2 Memory traffic — hard, and negligible (B0)

The irreducible HBM set per step is: the parameters the step actually reads,
plus the graph's inputs and outputs. Everything else is intermediate and fits
in L2.

```
P  total model params 9,984,957 (38.09 MiB fp32)
P    diffusion_stack 6,940,200 (26.47 MiB)   aa_decoder 1,867,204 (7.12 MiB)
P    prepare_mlp       949,376 ( 3.62 MiB)   angle_head   139,278 (0.53 MiB)
P  LEAN-STEP live params ~= 7,890,344 (30.10 MiB); dead on a lean step 7.97 MiB
B  graph_input_bytes = 1.401 MiB  graph_output_bytes = 0.209 MiB
```

Measured bandwidth (`x_micro.py` M3, graph-replayed so the number is the GPU's,
not the dispatcher's):

```
M3 add  10.00 MiB (ws  30 MiB)   4.816 us   6531.8 GB/s (3N)  8709.1 GB/s (4N)   <- L2-resident
M3 add  46.00 MiB (ws 138 MiB)  69.839 us   2072.0 GB/s (3N)  2762.6 GB/s (4N)   <- HBM
M3 add 256.00 MiB (ws 768 MiB) 365.769 us   2201.7 GB/s (3N)  2935.6 GB/s (4N)
```

31.7 MiB at the sustained HBM figure (2.94 TB/s counting the write-allocate
read) is **0.011 ms**; at the conservative 3N accounting, 0.015 ms. Even
re-reading the entire 38 MiB parameter set every step costs 0.014 ms.

That is the *external* traffic. It is not the whole memory term, and the
distinction matters — see immediately below.

**This kills the "0.37 ms memory roofline" that N007/N009 carried as a floor.**
That figure came from a whole-model *unfused* traffic census. The unfused census
is real — `x_census.py` measures **10.27 GiB** of input+output bytes across the
10,075 eager aten ops of a lean step — but it is an **upper** bound on traffic
(what a no-fusion, no-cache implementation would move), not a lower one.

### 2.2b Intermediate traffic — the term that nearly binds

Under *perfect* pointwise fusion a tensor still has to cross the memory
hierarchy if a barrier op produces or consumes it: a GEMM cannot read its
operand out of the producer's registers. Summing the input and output bytes of
every barrier node in the traced graph, each tensor counted once (job 904016):

```
D MIN-MATERIALISED-TRAFFIC (barrier-op inputs+outputs, counted once): 926.9 MiB
   -> 0.1488 ms at the measured 6.53 TB/s L2 rate
   -> 0.3306 ms at the measured 2.94 TB/s HBM rate
```

**927 MiB per step is irreducible for any kernel-level implementation of this
dataflow**, and where it lands decides the term. L2 is 60 MiB and the live set
at any stage is a few MiB (the largest single tensor is the `(6400,400)` dist
feature, 10.2 MiB, written and immediately consumed), so the overwhelming
majority should be L2 hits: **0.149 ms**. If it were all HBM it would be
0.331 ms and memory, not serialisation, would be the binding term.

So the three terms at L=100 are **0.150 (arithmetic) / 0.149 (L2 traffic) /
0.239–0.258 (serialisation)** — all the same order. The bound is therefore
reasonably *tight*, not a vacuous one: no single term dominates by more than
1.7×, which is unusual and worth noting, because it means there is no one thing
to fix at the floor.

### 2.3 Serialisation — hard, and it is the one that binds (B1)

A global barrier is forced whenever an operation's output element depends on
input elements the producer wrote from other thread blocks. Those barriers on
the **critical path** cannot be removed by any amount of fusion, batching or
reordering, because they are a property of the dataflow, not of the code.

`x_census.py` §D traces the aten graph of the shipping lean `forward_core`
through `aot_autograd` and computes the longest chain, counting **only** ops
that provably force a barrier (`mm/bmm/addmm`, `softmax`, `sum/mean/var/amax`,
`sort/argsort/topk`, `scatter_add`, `index_put`). Everything else — pointwise,
views, permutes, `cat`, and **gathers**, which inductor fuses into their
consumers — is counted as free, which is generous to the implementation and
therefore safe for a lower bound.

```
D aten_call_nodes=9448  barrier_nodes=779  CRITICAL_PATH_barriers=205
D level_widths(min/med/max)=6/19/2064  levels=206
D barrier op histogram: bmm=303, sum=251, mean=75, var=75, scatter_add=24,
                        sort=20, index_put=19, _softmax=12
```

The chain is a clean six-fold repeat of a 34-barrier block:

```
bmm -> sum -> bmm -> mean -> bmm -> bmm -> bmm -> _softmax -> bmm -> bmm ->
sum -> bmm -> mean -> mean -> bmm -> mean -> bmm -> _softmax -> bmm -> bmm ->
sum -> bmm -> mean -> bmm -> scatter_add -> bmm -> mean -> bmm -> bmm -> sum ->
index_put -> sort -> index_put -> sort
```

i.e. per block: pair features -> IPA#1 -> pair features -> IPA#2 -> global
update (`scatter_add` = `index_mean`) -> position update -> next block's
neighbour selection (two stable argsorts). 6 x 34 + 1 = **205**.

**What a barrier costs, measured two ways (`x_micro.py` M1, `x_micro2.py` M6):**

```
M1 tiny_1elem  streams=1 n=128..2048   marginal = 1.163 .. 1.167 us / kernel
M1 tiny_1elem  streams=8 n=128..2048   marginal = 0.187 .. 0.210 us / kernel
M6 grid barrier, persistent kernel, busy-wait:  1.482 us (1 block)
                                                1.882 us (132 blocks)
                                                2.044 us (264 blocks)
```

Two things follow.

1. **An empty kernel replayed from inside a captured CUDA graph, in a
   dependency chain, costs 1.166 µs — flat from 128 to 2048 kernels.** This is
   the per-launch term CUDA graphs *cannot* remove, measured rather than
   inferred.
2. **A hand-written grid-wide barrier inside one persistent kernel costs
   1.48–1.88 µs, i.e. MORE than a graph kernel launch.** So the standard
   escape hatch — "fuse the whole model into one mega-kernel and sync with
   cooperative groups" — does not lower the bound on this hardware. B1's
   synchronisation term is therefore also B0's: there is no cheaper primitive.

```
serialisation floor = 205 barriers x 1.166 us = 0.2390 ms
```

### 2.4 The bound

```
                  matmul  non-matmul  external  intermediate  serialisation
                  arith   arith       HBM       traffic (L2)  (205 barriers)
T_exact >= max(   0.150 , 0.005 ,     0.011 ,   0.149 ,       0.239   ) = 0.239 ms
T_fast  >= max(   0.020 , 0.005 ,     0.011 ,   0.149 ,       0.239   ) = 0.239 ms
```

Serialisation binds in both cases, because neither reduced precision nor a
smaller neighbourhood changes the depth of the graph. §6 tightens `exact`'s to
**0.258 ms** by charging each barrier stage its own arithmetic instead of a flat
launch cost; `fast`'s stays at 0.239.

---

## 3. Where the 3.57 ms actually goes

Kernel census of the shipping compiled forward (`x_census.py` §K, profiled
uncaptured so individual kernels are visible; the captured replay of the same
work is 3.395 ms — jobs 903967 and 903990 agree to 0.008 ms):

```
K preset=exact need_aa=False  kernels=946  gpu_busy = 3.5162 ms
K   GEMM (cuBLAS)   359 kernels  2.1602 ms   6.02 us each
K   Triton          524 kernels  1.2692 ms   2.42 us each
K   memcpy/memset    61 kernels  0.0819 ms   1.34 us each
K   HIST kernels=946  floor_term = 1.1030 ms   above_floor = 2.4132 ms
K     1.2- 2 us   96 kernels  0.1439 ms
K     2.0- 4 us  601 kernels  1.5300 ms
K     4.0- 8 us  206 kernels  1.2950 ms
K     8.0-16 us   43 kernels  0.5474 ms
```

**No kernel in the forward averages below 1.2 µs** — the histogram's first bin
is empty, which is the model's own confirmation of the 1.166 µs launch floor.
(The profiler aggregates by kernel *name*, so this is a per-name mean, not a
per-launch minimum; individual launches of a 2.4 µs-mean kernel could dip
below. It is still the right statement for the floor-term arithmetic, which
uses means.)

Decompose the measured 3.573 ms wall against it:

| term | ms | class |
|---|---|---|
| **serialisation: the weighted critical path (§6)** | **0.258** | **HARD** |
| excess launches: (946 − 205) kernels × 1.166 µs | 0.864 | soft |
| what the kernels spend above the launch floor, net of the arithmetic already inside the critical path | 2.394 | soft |
| python wrapper (outside the graph) | 0.178 | soft |
| reconciliation: captured replay is 0.121 ms under the uncaptured profile | −0.121 | — |
| **total** | **3.573** = measured | |

**0.258 ms of a 3.573 ms step is hard; 3.32 ms is implementation.** Of that
3.32 ms, §7 shows roughly 1.9 ms is the GPU sitting idle rather than the
kernels being wrong.

---

## 4. Attacks on the bound, and why each fails (measured, not argued)

A bound is only worth as much as the escapes it has closed. These are the four
a red team would try.

### 4.1 "Fuse the model into one persistent kernel and use grid barriers."

Measured: a grid barrier costs **more** than a graph kernel launch
(`x_micro2.py` M6, busy-wait variant, job 903989):

```
M6 blocks=1    1.482 us      M6 blocks=66   1.862 us
M6 blocks=8    1.830 us      M6 blocks=132  1.882 us
M6 blocks=32   1.853 us      M6 blocks=264  2.044 us
```

against `M1 streams=1 marginal = 1.166 us` for a kernel launch inside a
captured graph. A mega-kernel would make the serialisation term
`205 x 1.882 = 0.386 ms`, i.e. **62 % worse**. Closed.

### 4.2 "Use fewer SMs so the barrier is cheap."

M6 sweeps block count: the barrier only falls from 1.88 µs to 1.48 µs going
from 132 blocks to **one**, while the arithmetic floor rises as `132/k`. With
per-SM fp32 = 66.91/132 = 0.507 TFLOP/s:

| SMs used | arithmetic floor | 205 barriers | bound |
|---|---|---|---|
| 132 | 0.150 ms | 0.386 ms (grid) / 0.239 ms (launch) | 0.239 ms |
| 32 | 0.617 ms | 0.380 ms | 0.617 ms |
| 8 | 2.47 ms | 0.375 ms | 2.47 ms |
| 1 | 19.7 ms | 0.304 ms | 19.7 ms |

Synchronisation is nearly flat in occupancy and arithmetic is not, so shrinking
the launch never pays. Closed.

### 4.3 "D = 205 is itself an implementation artefact."

Partly true, and this is the **weakest term in the bound** — stated plainly
because the last floor claim died of not saying this. Three of the 34 barriers
per block are collapsible in principle:

* `mean -> mean` (LayerNorm's mean and variance) is one Welford kernel: −1
* `bmm -> _softmax -> bmm` twice per block is a flash-attention-shaped fusion:
  −4
* the trailing `index_put -> sort -> index_put -> sort` neighbour selection is
  two chained selection stages that a single kernel could in principle do: −2

That is 34 → 27 per block, `D_min ≈ 163`, serialisation term **0.190 ms**. §10
R1 pushes the argument as far as it will go (`D >= ~90`, 0.105 ms). Either way
the conclusion — a sub-millisecond step is not forbidden by this model's
dataflow — is unchanged, and no fusion argument reaches the current 3.4 ms.

### 4.4 "Overlap the steps."

The sampler is a Markov chain: step *t+1* consumes step *t*'s `pos` and
`local`. There is no step-level pipelining. Overlapping *designs* is a
throughput win, not a latency one — see §7.

---

## 5. `fast`: what each relaxation is worth AGAINST the bound

`fast` is allowed to diverge numerically, so the interesting question is which
relaxation moves the **bound** and which only closes the **gap**.

| relaxation | term it attacks | effect on the per-step BOUND | measured effect on the step |
|---|---|---|---|
| **TF32** (shipping) | arithmetic 0.150 → 0.020 ms | **none** — arithmetic was never binding | **−0.566 ms** (3.573 → 3.007) |
| **steps_scale=0.5** (shipping) | number of steps | per step none; **per design 0.104 → 0.048 s** | −50 % per design |
| K 64 → 32 both branches | arithmetic ~2×, traffic ~2× | none | −10 % (N011 §3) |
| bf16 GEMM (`medium`) | arithmetic, traffic | none | **+3 %** — slower than TF32 |
| shared self-conditioning draw | one `(N,N)` Gumbel | none | 0 (N011 §5) |
| P8 custom Triton pair kernel | **kernel count** — the 0.864 ms excess-launch term and part of the 2.39 ms under-utilisation | none (D is unchanged) | est. −1.2…−1.6 ms, **never built** |
| **trunk depth 6 → 3** | **D itself: 205 → 103** | **0.239 → 0.120 ms** | untested; changes the model |

Read the first row carefully. **TF32 buys 0.566 ms while the entire arithmetic
floor it relaxes is 0.150 ms.** So ~0.43 ms of TF32's win is not arithmetic at
all — it is that cuBLAS picks tensor-core kernels with a lower *per-kernel*
cost at these shapes. `x_micro.py` M4 shows exactly where:

```
M4 highest pair1.dist (6400,400)x(400,64)  16.188 us      M4 high  7.377 us   (2.19x)
M4 highest pair1.mlp1 (6400, 64)x( 64,128)  6.620 us      M4 high  4.834 us   (1.37x)
M4 highest pair2.rot  (4800,  9)x(  9,64)   3.147 us      M4 high  3.187 us   (0.99x)
M4 highest update.lin ( 100,128)x(128,512)  4.794 us      M4 high  4.252 us   (1.13x)
```

TF32 helps the five wide projections per block and does **nothing** for the
narrow ones, which are pure fixed cost. That is a complete explanation of the
16–19 % figure N011 measured, and it predicts — correctly — that bf16 adds
nothing on top (same tensor-core path, same fixed cost).

**Conclusion for `fast`: every knob it currently turns, and every knob it was
offered and refused, attacks a term that is not binding.** Its per-step floor
is identical to `exact`'s. The only lever that would lower `fast`'s per-step
bound is reducing the trunk's serial depth — distillation to fewer blocks —
and that is a model change, not an inference preset.

---

## 6. The bound, refined: a weighted critical path

`D x c_launch` assumes every barrier stage costs exactly the empty-kernel
floor. The wide projections cost more than that even at peak. `x_census.py` §D
therefore also computes the longest chain under the node weight
`max(c_launch, node_FLOPs / peak)`, with FLOPs annotated from the traced
graph's fake-tensor metadata (job 903996, **303/303 matmul nodes annotated**):

```
D weighted_critical_path_fp32 = 0.2581 ms
D weighted_critical_path_tf32 = 0.2390 ms
D graph_matmul_flops = 10.0122 GFLOP -> 0.1496 ms at fp32 peak, 0.0202 ms at tf32 peak
```

**Final bounds:**

| | lower bound, per lean step | measured | factor |
|---|---|---|---|
| `exact` (fp32) | **0.258 ms** | 3.573 ms | **13.8×** |
| `fast` (TF32) | **0.239 ms** | 3.007 ms | **12.6×** |

The two differ by 7 %. Under TF32 no single matmul on the critical path costs
more than the launch floor, so `fast`'s bound collapses exactly onto
`D x c_launch`.

---

## 7. The capacity floor: what the CURRENT kernels cost when the GPU is busy

The bound above is a bound on the *problem*. The other number worth having is a
bound on *this kernel decomposition*: what these 946 kernels cost when they are
not waiting for each other. Measure it by saturating the GPU — capture *k*
independent copies of the compiled forward into one graph, one per stream
(`x_concurrency.py`, job 903998):

```
M9 exact L=100 k=1  total= 3.6961 ms  per_forward=3.6961 ms  1.00x
M9 exact L=100 k=2  total= 5.0863 ms  per_forward=2.5431 ms  1.45x
M9 exact L=100 k=4  total= 7.6237 ms  per_forward=1.9059 ms  1.94x
M9 exact L=100 k=8  total=13.3064 ms  per_forward=1.6633 ms  2.22x
M9 fast  L=100 k=8  total=11.1161 ms  per_forward=1.3895 ms  2.26x
```

**One forward uses under half of what the H100 can deliver to these kernels.**
Eight concurrent forwards cost 3.6× one forward, not 8×. So:

| | today | capacity floor (same kernels, GPU saturated) | bound (any kernels) |
|---|---|---|---|
| `exact` lean step | 3.573 ms | **~1.66 ms** | 0.258 ms |
| `fast` lean step | 3.007 ms | **~1.39 ms** | 0.239 ms |

This splits the gap into two regimes with completely different remedies:

* **3.57 → 1.66 ms is scheduling.** The kernels are fine; the GPU is idle
  waiting for them one at a time. Fixed by overlap, not by fusion.
* **1.66 → 0.26 ms needs different kernels** — fewer and bigger. This is the
  P8 territory, and it is the only route past the capacity floor.

Two immediate consequences, one of which is the largest unclaimed win in the
whole speed effort:

**Throughput, available today, no numerical change at all.** Designs are
independent. Capturing 8 designs' denoiser steps into one graph is **2.22×
(exact) / 2.26× (fast) more designs per second**, because each forward is
byte-for-byte the same computation it is now, merely scheduled alongside
others. At L=100 that takes `fast` from 0.58 s/design to **0.26 s/design** and
`exact` from 1.43 s to 0.64 s. Nothing in N007/N009/N011 considered this: every
proposal so far optimised one forward in isolation, which is the one regime
where the GPU is 55 % idle.

**Latency**, the harder half, needs the forward's *own* independent work
overlapped — the graph is wide (`level_widths` median 19, max 2064 over 206
levels), so the parallelism is there; it is the single-stream stream-capture
that serialises it. Untested, and the honest statement is that M9 bounds what
it could buy (≈ 3.57 → ~1.7 ms) without showing it is achievable.

---

## 8. Ranked gap list — `exact` (3.573 ms wall → 0.258 ms bound)

Written as a **ladder**, not a sum, because several items attack the same
milliseconds and adding them would be the N009 mistake in a new costume.

| # | item | worth | class | status / reachability |
|---|---|---|---|---|
| **E1** | **GPU idle: one forward saturates 45 % of the H100** | **3.57 → ~1.66 ms** (−1.91) | soft, scheduling | **throughput form available today** (8-design graph, 2.22×, zero numerical change). Latency form (multi-stream capture of the forward's own independent work) untried. |
| E2 | 741 kernels beyond the 205 on the critical path, each paying the 1.166 µs launch floor | 0.864 ms of floor | soft | fusion/batching; overlaps E1 |
| E3 | python wrapper outside the graph | 0.178 ms (5.0 %) | soft | capture *N* sampler steps in one graph; the sampler's own arithmetic is capturable, the `uniform` draws are already hoisted |
| E4 | narrow projections on cuBLAS rather than a Triton template | ~0.12 ms | soft | **measured and bit-exact at fp32** — see the E4/F4 detail below |
| E5 | condition-branch GEMMs not batched across the 6 blocks (P7) | ≤0.167 ms of GEMM time | soft | **in flight (N014)**; this item only sizes it |
| E6 | 524 Triton kernels at 2.42 µs against a 1.166 µs floor | 0.66 ms | soft | **in flight (N017)**; `aggressive_fusion` was −1.7 % |
| **H1** | **serialisation: 205 critical-path barriers × 1.166 µs, weighted by peak arithmetic** | **0.258 ms** | **HARD** | irreducible without changing trunk depth |
| H2 | arithmetic at fp32 peak (already inside H1) | 0.150 ms | HARD | TF32 relaxes it to 0.020 ms — and that is `fast` |
| H3 | external HBM traffic (31.7 MiB params + I/O at 2.94 TB/s) | 0.011 ms | HARD | 0.3 % of the step |
| H4 | irreducible intermediate traffic (926.9 MiB across barrier ops) | 0.149 ms at L2 rate, 0.331 ms if it spills to HBM | HARD | the term nearest to displacing H1 — see §2.2b and §10/R2 |

## 9. Ranked gap list — `fast` (3.007 ms wall → 0.239 ms bound)

Identical structure; `fast` has one extra lever and one fewer term.

| # | item | worth | class | status |
|---|---|---|---|---|
| **F1** | GPU idle | 3.01 → ~1.39 ms (−1.62) | soft | as E1; **2.26× throughput today → 0.26 s/design** |
| F2 | excess launches beyond D | ~0.86 ms of floor | soft | as E2 |
| F3 | wrapper | 0.172 ms | soft | as E3 |
| F4 | narrow projections: Triton beats cuBLAS 1.29–1.43× at TF32 too | ~0.10 ms | soft | detail below; **not** bit-exact at TF32, but `fast` does not require it |
| F5 | P8 custom fused pair kernel | est. −1.2…−1.6 ms | soft | **never built**; the only route past the ~1.39 ms capacity floor |
| F6 | trunk depth 6 → 3 (distillation) | **bound 0.239 → 0.120 ms** | model change | the ONLY lever that lowers `fast`'s bound |
| H1' | serialisation | 0.239 ms | HARD | — |
| H2' | arithmetic at TF32 peak | 0.020 ms | HARD | 0.7 % of the step |
| H3' | irreducible intermediate traffic | 0.149 ms (L2) | HARD | identical to `exact`'s — precision does not change tensor counts, only their width, and both presets keep fp32 tensors |

### E4 / F4 in full — cuBLAS's fixed cost is a library choice, not hardware

`x_triton_gemm.py` (job 903996) compares cuBLAS against an inductor Triton
template at the model's real shapes, both graph-replayed:

```
M8 highest pair2.rot   (4800,  9)x(  9,64)  cublas  3.182  triton  2.322  1.37x  max|d| 0.000e+00
M8 highest pair1.rot   (6400,  9)x(  9,64)  cublas  3.352  triton  2.549  1.31x  max|d| 0.000e+00
M8 highest attn2.bias  (4800, 64)x( 64,  8) cublas  4.127  triton  3.192  1.29x  max|d| 0.000e+00
M8 highest pair2.dir   (4800, 60)x( 60,64)  cublas  4.750  triton  3.930  1.21x  max|d| 0.000e+00
M8 highest update.lin  ( 100,128)x(128,512) cublas  4.819  triton  3.537  1.36x  max|d| 0.000e+00
M8 highest proj_out    ( 100,1024)x(1024,128) cublas 7.221  triton 14.951  0.48x  max|d| 1.602e-04
M8 highest pair2.dist  (4800,400)x(400,64)  cublas 13.387  triton 15.168  0.88x  max|d| 9.155e-05
```

The narrow projections — which are most of the 359 cuBLAS launches — are
**1.21–1.37× faster on Triton and BIT-IDENTICAL at fp32** (`max|d| = 0.0`,
because K ≤ 128 fits a single-pass reduction and the accumulation order is
unchanged). The wide ones (`dist`, `proj_out`) are slower on Triton and must
stay on cuBLAS. So the right move is per-shape selection, worth ≈0.12 ms and
costing **nothing** in parity — the only such item on either list.

This also disposes of N007's "the `dist` projection runs at ~51 TFLOP/s, ~76 %
of fp32 peak, there is nothing left inside it". Measured properly (inside a
graph, so the number is the kernel's and not the dispatcher's),
`(6400,400)x(400,64)` costs **16.19 µs = 20.2 TFLOP/s = 30 % of peak**, not
76 %. The 51 TFLOP/s figure came from an eager-loop timing where ~6–8 µs of CPU
dispatch was being counted as GPU work.

---

## 9b. Where TF32's 0.58 ms actually comes from (kernel census, both presets)

```
K exact lean  946 kernels 3.5162 ms   cuBLAS 359 @ 6.02 us = 2.1602   Triton 524 @ 2.42 = 1.2692
K fast  lean  934 kernels 2.9302 ms   cuBLAS 347 @ 4.57 us = 1.5841   Triton 524 @ 2.40 = 1.2597
```

TF32 changes the Triton stage by **0.010 ms** and the cuBLAS stage by
**0.576 ms**, entirely as a lower cost *per GEMM launch* (6.02 → 4.57 µs). The
arithmetic floor it relaxes is 0.130 ms, so **78 % of TF32's win is not
arithmetic** — it is cuBLAS selecting a cheaper tensor-core kernel at shapes
that are latency-bound either way. That is why bf16 adds nothing on top (same
path, same fixed cost) and why shrinking K buys only 10 %.

Neither preset has a kernel *name* averaging below 1.2 µs: both duration
histograms start at the 1.166 µs floor and the median kernel sits in the 2–4 µs
bin (exact 601/946, fast 704/934).

## 9c. Length dependence — the occupancy gap is an L=100 phenomenon

```
M9 exact L=100  k=1 3.6961 ms   k=4 1.9059/fwd (1.94x)   k=8 1.6633/fwd (2.22x)
M9 exact L=400  k=1 7.4461 ms   k=4 5.8424/fwd (1.27x)
```

At L=400 the model already fills the GPU, and 4-way concurrency buys only
1.27×. So the §7 result is specifically that **short proteins under-use the
H100**, which is exactly the regime `--num_aa 100` lives in.

The critical-path depth is **exactly length-independent** — job 904007 re-traced
the graph at L=400 and got `CRITICAL_PATH_barriers=205`, the same 205, with the
same `level_widths 6/19/2064`. Only the arithmetic grows:

| | L=100 | L=400 |
|---|---|---|
| matmul FLOPs | 10.01 GFLOP | 40.05 GFLOP |
| arithmetic at fp32 peak | 0.150 ms | **0.599 ms** |
| arithmetic at TF32 peak | 0.020 ms | 0.081 ms |
| weighted critical path, fp32 | **0.258 ms** | 0.408 ms |
| weighted critical path, TF32 | **0.239 ms** | **0.241 ms** |
| **bound, `exact`** | **0.258 ms** | **0.599 ms** (arithmetic binds) |
| **bound, `fast`** | **0.239 ms** | **0.241 ms** |
| measured lean step | 3.573 / 3.007 ms | 7.38 / 6.69 ms (N009/N011) |
| factor above bound | 13.8× / 12.6× | 12.3× / 27.8× |

The two presets' bounds are equal at L=100 and **diverge by 2.5× at L=400**,
because past the crossover (measured interpolation: **L ≈ 208**) fp32
arithmetic becomes the binding term while TF32 arithmetic stays far below the
serialisation floor. That is the one place where `fast`'s licence buys real
floor rather than only slack.

---

## 10. What a red team can still break — stated before they find it

Ranked by how much of the bound moves if the objection lands.

**R1 — `D = 205` is a classification, not a theorem.** *(biggest exposure)*
I count a barrier wherever an op's output element depends on input elements
another thread block wrote. Cross-op tiling defeats that in specific cases:
flash attention fuses `bmm -> softmax -> bmm` into one kernel, Welford fuses
`mean -> var`, a single kernel could in principle do both neighbour selection
stages, and — this is P8 exactly — a kernel that computes the pair features in
registers collapses each `sum -> bmm` (distance reduction feeding the 400-wide
projection) into one. §4.3 prices the first three at `D = 163` → **0.190 ms**;
adding the four `sum -> bmm` collapses per block takes it to `D ≈ 139` →
**0.162 ms**. A more
aggressive attack could push further, but not indefinitely: the chain
`LN -> qkv -> dot -> softmax -> out -> project_out -> LN -> …` is genuinely
serial across 6 blocks, so `D >= ~90` and the bound never goes below
**~0.105 ms** on any argument I can construct. **Range: 0.105–0.258 ms.**
Everything else in this document survives anywhere in that range.

**R2 — the memory term hinges on L2 hit rate, which I bounded but did not
measure.** §2.2b measures the irreducible *materialised* traffic at
**926.9 MiB/step**. At the measured L2 rate (6.53 TB/s) that is 0.149 ms and
does not bind; at the measured HBM rate (2.94 TB/s) it is **0.331 ms and it
does bind**, which would raise the bound from 0.258 to ~0.33 ms. The physical
argument for L2 is strong (60 MiB of L2, largest single tensor 10.2 MiB,
produced and consumed adjacently) but it is an argument, not a measurement. An
Nsight Compute run (`dram__bytes_read.sum` + `dram__bytes_write.sum`,
NVTX-scoped to one lean forward) is in `x_ncu.py` / `run_r2b.sbatch`; see the
log for whether it landed. **This is the term I would attack first if I were
the red team**, and the worst case for me moves the bound by 1.3×, not by an
order of magnitude — the headline (13.8× above the floor) survives it.

**R3 — `c_launch = 1.166 µs` is this driver on this node.** It is the marginal
replay cost of a 1-element `add_` in a captured graph, stable over 128–2048
kernels and measured twice. Device-side graph launch (CUDA 12 `cudaGraphLaunch`
from device code) was not tried and could be cheaper; a newer driver could
change it. The grid-barrier alternative was measured and is *worse* (§4.1), so
1.166 µs is the best primitive I found, not provably the best that exists.

**R4 — the 2.22× concurrency result is measured on the compiled core, not on
`generate.py`.** `x_concurrency.py` captures *k* copies of
`Diffusion.forward_core` with independent inputs. A real 8-design sampler adds
per-step host work, 8× the activation footprint, and the conditioning-cache
path. The *number* is solid; the claim "0.26 s/design" is a projection until
someone builds the batched sampler. Also note `M9 k=1 = 3.696 ms` against the
production replay's 3.395 ms — the harness differs (fresh argument sets,
graph-pool allocation), so I use the **ratios**, not the absolute k=1 value.

**R5 — L=100 is the headline; L=400 is checked, nothing between is.** Job
904007 re-traced at L=400: `D = 205` exactly, so the serialisation term is
length-independent, and the crossover to an arithmetic-bound regime for `exact`
lands near **L ≈ 208** (§9c). Concurrency headroom falls from 2.22× at L=100 to
1.27× at L=400. L=200, the other common length, was not measured and sits right
on the crossover.

**R6 — `fast`'s arithmetic floor uses NVIDIA's 494.7 TFLOP/s TF32 dense spec,
not a measurement.** Best I actually measured is 77.2 TFLOP/s (`pairvec`
batched-by-6). Even at 1/6 of spec the TF32 arithmetic floor is 0.12 ms, still
under the 0.239 ms serialisation term, so the conclusion does not move.

**R7 — FLOPs were matmul-only. NOW MEASURED, and it does not bind.**
`FlopCounterMode` counts matmuls only, so I counted the rest directly off the
traced graph: every non-matmul, non-view, non-gather node's output elements,
charged 1 FLOP each and 4 for transcendentals/`rsqrt`/`pow` (SFU rate). Job
904012:

```
D NON-MATMUL arithmetic: 228.53 M output elements, 0.3484 GFLOP charged
                         -> 0.0052 ms at fp32 peak
```

**0.005 ms**, i.e. 2 % of the serialisation term. The elementwise work is not
an arithmetic problem — it is a *traffic* problem, and that is R2, not this.
My own pre-measurement guess ("~1.3 G elements, could double the floor") came
from the eager dispatch tally, which double-counts every view and gather; the
graph-level count is the right one.

**R8 — the ~1.66 ms "capacity floor" is an empirical saturation point, not a
bound.** It is `k=8` on the curve 1.00 / 1.45 / 1.94 / 2.22×, which is still
rising; `k=16` was not tried and would land it lower. It also holds the current
kernel decomposition fixed, so it is a **B2** number in §1's taxonomy and says
nothing about the problem. And the 8 forwards share one copy of the weights in
L2 — which flatters the number, but is also exactly what a batched sampler
would do, so it is representative rather than optimistic *for that use*.

**R9 — everything is one checkpoint, one config.** `default_vp-200k`, no
cyclic, no multimotif, no pair conditioning (`pair_cond=False`, the de-novo
path). `multimotif_vp` routes through `MotifAdapterDiffusionBlock`, whose
second pair branch reads `pos` rather than `prev_pos`; its graph was not
traced, so `D = 205` is not asserted for it.

**Not contested:** the measured baselines (3.573 / 3.007 ms), the kernel census
(946 kernels, 3.52 ms), the 1.166 µs launch floor, the 1.48–2.04 µs grid
barrier sweep, the Triton-vs-cuBLAS table, and the concurrency ratios. Those
are direct measurements with their harnesses in this directory.

---

## 11. Honest summary

| | `exact` | `fast` |
|---|---|---|
| measured lean step, L=100 | 3.573 ms | 3.007 ms |
| **bound (headline)** | **0.258 ms** | **0.239 ms** |
| worst case if R2 lands (all 927 MiB goes to HBM) | ~0.33 ms | ~0.33 ms |
| best case if R1 lands (maximal cross-op fusion, D≈90) | ~0.15 ms | ~0.11 ms |
| **defensible range** | **0.15 – 0.33 ms** | **0.11 – 0.33 ms** |
| capacity floor of today's kernels | ~1.66 ms | ~1.39 ms |
| bound at L=400 | 0.599 ms | 0.241 ms |

Even at the pessimistic end of the range the model is **>10× above its floor**
on both presets, which is the claim that has to survive; the headline number is
0.26 ms and the honest interval is 0.15–0.33 ms.

Three sentences:

1. **The floor is a serialisation floor, not a FLOP or bandwidth floor.** At
   L=100 matmul arithmetic is 0.150 ms, non-matmul arithmetic 0.005 ms,
   external HBM traffic 0.011 ms and irreducible intermediate traffic 0.149 ms
   (L2-resident); the 205 unavoidable global barriers on the critical path are
   0.24–0.26 ms, and everything above that is implementation. All four terms
   are the same order, so the bound is tight rather than vacuous.
2. **`exact` and `fast` have the same per-step floor** (within 7 %), because
   every relaxation `fast` is permitted attacks a term that was never binding.
   `fast`'s win is per-design (half the steps) and per-kernel-cost (tensor-core
   cuBLAS), not per-floor.
3. **The largest unclaimed win is not fusion, it is occupancy**: one forward
   uses 45 % of what the H100 delivers to these kernels, and eight concurrent
   forwards cost 3.6× one — a 2.2× throughput gain available today with a
   byte-identical computation.

---

## 12. Files

| file | what |
|---|---|
| `x_micro.py` | DEV / M1 empty-graph replay vs kernel count / M3 bandwidth / M4 GEMM at model shapes — **all graph-replayed** |
| `x_micro2.py` | M5 multi-stream overlap at model shapes / M6 grid-barrier sweep |
| `x_census.py` | P params, F FLOPs, B traffic, D critical path (+ weighted), K kernel census with a duration histogram |
| `x_baseline.py` | production wall time split into captured replay + python wrapper |
| `x_triton_gemm.py` | M8 cuBLAS vs inductor Triton template at the model's shapes |
| `x_concurrency.py` | M9 *k* independent forwards in one graph on *k* streams |
| `run_*.sbatch`, `logs/` | one process per configuration, one length per process |

Jobs: 903966 (void — eager-loop timing, see §9), 903967, 903971, 903977,
903979, 903980, 903989, 903990, 903996, 903998, 904007. Each one H100 on the
`h100` partition, all well under 1 GPU-hour; total for the item < 2 GPU-h.
GPU was required throughout — the entire question is CUDA kernel launch and
occupancy behaviour, which has no CPU analogue.

**Method note that cost two jobs and is worth carrying forward:** timing a
kernel by looping it in eager mode and wrapping CUDA events around the loop
measures the **~6–8 µs CPU dispatch cost**, not the kernel. It produces
plausible, stable, wrong numbers — every small GEMM comes out at "8.0 µs"
regardless of shape. Job 903966 did this; job 903977 redid everything inside a
captured CUDA graph and the same GEMMs came out at 3.1–16.2 µs. N007's
"51 TFLOP/s, 76 % of peak" for the `dist` projection is an instance of the same
error.
