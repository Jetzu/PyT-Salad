# N029 — where the 4.6 GiB of L2 traffic goes, and how much of it is real

All numbers: H100 (132 SM, sm_90, 50 MiB L2, 94 GiB), nodes h11-23/24 + h13-11,
torch 2.5.1+cu121, `default_vp-200k`, **L=100**, lean step (`need_aa=False`,
`pair_cond=False`), preset `exact`, one length per process, detached worktree
`/hnvme/workspace/v181bb10-salad/n029-l2` at HEAD `98bc61b`.
Jobs 907096 (graph), 907097 + 907131 (Nsight Compute), 907101 (inductor
`output_code`), 907140 (kernel-time census), 907141 (`fast`/TF32 ncu),
907152 (O1 A/B), 907154 (CPU bitwise control).

Baseline reproduced in job 907096: **lean 3.393 ms / full 4.186 ms at L=100**
(the brief's 3.375 ms).

---

## 0. The verdict

**E7 closes as a negative result. The "5.2x over the minimum" is not 5.2x of
fusion headroom; it is a ratio between two quantities that were never the same
kind of thing.** Decomposed, measured:

| factor | what it is | value | reducible? |
|---|---|---|---|
| A | N016's graph-derived minimum: distinct barrier tensors, **each counted once** | 926.9 MiB | — (definition) |
| ×1.75 | every materialised tensor must be **written and read**, not touched once | → 1618.5 MiB logical | **no** — arithmetic identity of any kernel-level implementation |
| ×2.08 | SM↔L2 access amplification (32-byte sectors, GEMM tile re-reads, gathers) | → 3372.1 MiB SM-side | partly, and only by changing layouts/kernels |
| ×1.36 | DRAM-side fill + writeback, which `lts__t_bytes` also counts | → **4602.4 MiB** measured | no |
| | **product** | **4.96x** | |

The load-bearing line is the second one. N016's 926.9 MiB counts each tensor
**once**; any implementation that materialises a tensor pays for it **twice**
(one store, at least one load). The true kernel-level floor for this dataflow is
therefore between 926.9 and 1853.8 MiB, and **the current implementation's
logical traffic is 1618.5 MiB — inside that band, at 0.87 of its top.**

Inductor is not leaving materialisation on the table. Counting every buffer the
generated code allocates and every kernel argument that reads it (job 907101's
`output_code`, parsed by `scripts/parse_code.py`): **771 MiB of stores and
847 MiB of loads**, against a 926.9 MiB "minimum" that counts the same tensors
once. There is no 3.9 GiB of un-fused intermediates to find.

**What fusion could still remove is 82.4 MiB.** Classifying every buffer by the
kind of kernel that produces and consumes it (`scripts/flow.py`):

| producer → consumer | logical read bytes | edges | can pointwise fusion remove it? |
|---|---|---|---|
| triton → **gemm** | 511.1 MiB | 391 | only with a fused-prologue matmul (no cuBLAS) |
| triton → triton, **same kernel** (`in_out_ptr` accumulate) | 87.6 MiB | — | already minimal |
| triton → triton, **cross-kernel** | **82.4 MiB** | — | **yes — this is the whole prize** |
| gemm → triton | 123.7 MiB | 300 | only with an epilogue (no cuBLAS) |
| param/input → anything | 37.5 MiB | 611 | no |
| gemm → gemm | 4.9 MiB | 24 | no |

82.4 MiB of reads plus the matching 82.4 MiB of writes is 164.8 MiB logical,
~343 MiB measured — **7.5 % of the 4.60 GiB, worth ~0.10 ms (3 %) of the
3.393 ms step** at the exchange rate measured in §3. That is the honest size of
the remaining pointwise-fusion prize, and it is nine times smaller than the
figure E7 implied.

**The one large addressable class is not fusion at all — it is layout.**
776.3 MiB (**17.0 % of all L2 traffic**) is moved by 67 Triton `clone` kernels
whose only job is to re-lay-out a gathered tensor so `extern_kernels.bmm` will
accept it. That traffic and those kernels exist because the IPA contractions are
written as `torch.einsum`, which ATen lowers to `bmm`. It is N027/N028's
einsum→mul+sum question (N017's O2), and this item sizes it: **776 MiB and 67
launches, ~0.30 ms, 8.8 % of the step**, the largest single quantified item in
the forward.

---

## 1. The measurement, and what changed from N016

Same harness as N016's `x_ncu.py` (uncaptured compiled call, NVTX range
`"measure/"`, `--print-summary per-gpu`, weighted by `Invocations`), re-run at
this HEAD with `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1` so every Triton group is a
named row instead of 57 anonymous `triton_`s.

| | N016 (a8533a7, P7 on) | **N029 (98bc61b, P7 off)** |
|---|---|---|
| kernels in the NVTX range | 885 | **746** |
| `lts__t_bytes.sum` | 4814.85 MiB | **4576.0 / 4602.4 MiB** (two jobs) |
| `dram__bytes_read.sum` | 890.94 MiB | **896.6 MiB** |
| L2 hit rate | 81.5 % | **80.4 %** |

Two runs of the identical configuration (907097, 907131) give 4576.0 and
4602.4 MiB, i.e. ±0.3 % run-to-run. N016's number reproduces.

### 1.1 The read/write/atomic split (job 907131)

| class | launches | `lts__t_bytes` | read | write | atomic | reduce | rd+wr+at+red |
|---|---|---|---|---|---|---|---|
| cuBLAS | 352 | 2332.5 | 1372.3 | 389.7 | 0.0 | 0.0 | 1761.9 |
| Triton | 392 | 2263.4 | 723.5 | 875.1 | 0.0 | 8.2 | 1606.8 |
| other | 2 | 6.5 | 1.2 | 2.2 | 0.0 | 0.0 | 3.4 |
| **total** | **746** | **4602.4** | **2097.0** | **1266.9** | **0.0** | **8.2** | **3372.1** |

(MiB; sector counts x 32 B.)

Two readings that matter:

* **Atomic traffic is exactly zero.** The `index_sum` `scatter_add` in `Update`
  was an obvious suspect for hidden traffic — 51 200 atomic adds onto 512
  addresses, twelve times a forward. It is not: `lts__t_sectors_op_atom.sum`
  is 0.00 for all 182 groups (inductor lowers that scatter to ordinary stores
  because the index is a broadcast, not a true scatter). Do not spend effort
  there.
* **1230.3 MiB of the 4602.4 (26.7 %) is not SM-side read/write at all.**
  `lts__t_bytes` counts every byte crossing the L2 slices; the four `op_*`
  sector counters (read, write, atom, red) sum to only 3372.1 MiB. The natural
  candidate for the residual is the DRAM-side fill and writeback traffic, which
  is the right order (896.6 MiB of measured DRAM reads plus an uncollected write
  stream — `dram__bytes_write.sum` returns 0.00 on this driver, as N016 found),
  though the per-class split does not match it cleanly (cuBLAS residual 570.6
  vs 622.8 MiB of DRAM reads; Triton 656.6 vs 273.0) and this node's counters
  cannot settle it. **Marked `[unconfirmed]` as to its origin; not marked
  uncertain as to its consequence** — it is proportional to traffic that misses
  L2 and no source change reaches it directly. **The SM-side figure,
  3372.1 MiB, is the one an optimisation can move**, and it is what §0's 2.08x
  amplification is measured against.

### 1.2 By kernel class

| class | launches | measured L2 | logical (from `output_code`) | **amplification** | time (job 907140) | effective L2 rate |
|---|---|---|---|---|---|---|
| cuBLAS | 352 | 2335.5 MiB | 735.0 MiB operands | **3.18x** | 2166.3 us | 1.13 TB/s (17 % of 6.53) |
| Triton | 392 | 2234.0 MiB | 883.4 MiB | **2.53x** | 1077.6 us | 2.17 TB/s (33 %) |
| memcpy | 61 | — | — | — | 82.6 us | — |
| **step** | **807** | **4576.0 MiB** | **1618.5 MiB** | **2.83x** | **3331.3 us** | **1.44 TB/s (22 %)** |

cuBLAS spends 2166 us on 352 launches — **6.15 us each, against N017's measured
5.98 us launch floor**. Essentially *all* cuBLAS time on this model is launch
latency, so **bytes on the GEMM side are free**: removing GEMM traffic buys
nothing unless it removes a launch. Triton, at 2.17 TB/s overall and 3.54 TB/s
net of its 415 us launch floor, is the only half where bytes convert to time.

### 1.3 TF32 moves traffic too (job 907141, preset `fast`)

| preset | cuBLAS L2 | cuBLAS L2 read | Triton L2 | total |
|---|---|---|---|---|
| `exact` (fp32 SIMT) | 2335.5 | 1374.8 | 2234.0 | 4576.0 MiB |
| `fast` (TF32 tensor core) | 1953.3 | 1084.9 | 2231.0 | 4190.9 MiB |
| delta | **−16.4 %** | **−21.1 %** | −0.1 % | **−8.4 %** |

The tensor-core kernels use larger tiles and therefore re-read their operands
less: the cuBLAS amplification falls 3.18x → 2.66x. Triton is untouched to
0.1 %, which is a clean internal control that the difference really is the GEMM
kernels and not the graph. **PORTABLE, and it matters more on a consumer card**
(~1 TB/s of L2 instead of 3.35, so the same bytes cost 3x the time).

---

## 2. The byte attribution

### 2.1 Top 16 kernel groups by measured L2 bytes (job 907097)

Grid shape in parentheses; `rd`/`wr` are the SM-side read/write split.

| MiB | % | inv | MiB/ea | rd | wr | kernel | what it is |
|---|---|---|---|---|---|---|---|
| 254.3 | 5.6 | 24 | 10.60 | 172.2 | 42.3 | `sm80_xmma_gemm …tilesize32x32x8` | the narrow fp32 projections |
| 249.8 | 5.5 | 12 | 20.82 | 171.3 | 56.2 | `sm80_xmma_gemm …tilesize64x64x8` | `linear_dist` (6400x400x64) |
| 222.2 | 4.9 | 24 | 9.26 | 180.7 | 3.0 | `gemv2N_kernel` | the batch-800 M=1 IPA contractions |
| **186.9** | **4.1** | 6 | 31.15 | 58.8 | 112.2 | **`triton_poi_fused_clone_20`** | **re-layout of `v[nb]`+`v_g[nb]` for bmm** |
| 172.5 | 3.8 | 18 | 9.58 | 99.7 | 42.4 | `sm80_xmma_gemm …32x32x8` | projections |
| 155.9 | 3.4 | 6 | 25.99 | 113.1 | 31.7 | `sm80_xmma_gemm …64x64x8` | `linear_pairvec` (6400x240x64) |
| 150.0 | 3.3 | 6 | 25.01 | 50.8 | 84.4 | `…add_clone_index_mean_mul_rsqrt_sub_var_47` | LayerNorm(k) + gather, IPA |
| 120.9 | 2.6 | 6 | 20.15 | 43.9 | 63.4 | `…clone_index_pow_sub_51` | the q_g/k_g point-distance term |
| 101.6 | 2.2 | 6 | 16.94 | 62.9 | 28.1 | `…add_index_mean_mul_rsqrt_sub_var_28` | LayerNorm of `pair` (N,K,64) |
| 99.4 | 2.2 | 6 | 16.58 | 74.9 | 14.0 | `sm80_xmma_gemm …64x64x8` | `project_out` (100x1024x128) |
| 97.6 | 2.1 | 30 | 3.25 | 40.9 | 9.3 | `sm80_xmma_gemm …32x32x8` | projections |
| 96.9 | 2.1 | 6 | 16.15 | 31.6 | 53.0 | `triton_poi_fused_cat_27` | `pair_vector_features`' (N,64,240) |
| **96.9** | **2.1** | 6 | 16.15 | 42.4 | 42.2 | **`triton_poi_fused_clone_48`** | **permute of gathered `k` for bmm** |
| 84.2 | 1.8 | 6 | 14.03 | 4.5 | 66.0 | `…arange_div_exp_mul_neg_pow_sub_40` | the 400-channel distance RBF |
| 81.4 | 1.8 | 6 | 13.56 | 48.6 | 22.4 | `…bitwise_and_cat_clamp…_46` | pair mask + relpos + pair LayerNorm |
| 73.8 | 1.6 | 6 | 12.30 | 22.5 | 39.6 | `triton_poi_fused_cat_45` | `pair_vector_features`' (N,48,240) |

The distribution is flat — the largest single group is 5.6 %, and it takes 16
groups to reach 45 %. There is no hot spot to delete; §0's classification is the
only way to reason about it.

### 2.2 Triton traffic by role

| role | measured L2 | % of the 4.58 GiB | logical | amp | launches |
|---|---|---|---|---|---|
| **`clone` — re-layout for `bmm`** | **776.3 MiB** | **17.0 %** | 311.7 | 2.49x | 67 |
| other pointwise | 516.3 | 11.3 % | 184.5 | 2.80x | 135 |
| `cat` — feature assembly | 441.0 | 9.6 % | 203.1 | 2.17x | 70 |
| LayerNorm | 166.3 | 3.6 % | 66.7 | 2.49x | 39 |
| gelu / MLP | 150.0 | 3.3 % | 75.4 | 1.99x | 25 |
| softmax | 143.5 | 3.1 % | 40.1 | 3.58x | 36 |
| neighbour `sort` | 40.6 | 0.9 % | 2.0 | 20.6x | 20 |
| **Triton total** | **2234.0** | **48.8 %** | **883.4** | **2.53x** | **392** |

### 2.3 Where the traffic lives in the model (traced ATEN graph, job 907096)

Unfused traffic = every non-view node's inputs+outputs; min = N016's barrier
rule. The two columns are the upper and lower bounds that bracket the real
implementation.

| source site | unfused | min | note |
|---|---|---|---|
| `features.py:43 gaussian_rbf` | 933.7 MiB (22.5 %) | **0.0** | entirely fusible — and inductor **does** fuse it: only the final (N,K,400) is materialised |
| `basic.py:94 HkLinear.forward` | 417.4 | **417.4** (45 % of the min) | the GEMM operands — irreducible without a custom matmul |
| `attention.py:108/109/120/121` | 1052.0 | 241.0 | IPA point/dot/value contractions |
| `rigid.py:43 vec_norm2` | 223.7 | 89.5 | |
| `frames.py:92 apply_inverse_t` | 189.5 | 94.8 | |
| `features.py:212-213 pair_vector_features` | 184.5 | **0.0** | fusible; see §4 |
| **total** | **4145.4 MiB** | **926.9 MiB** | |

Note the shape of this: the *unfused* graph traffic (4145 MiB) is close to the
*measured* L2 traffic (4576 MiB), which invites the reading "nothing is fused".
That reading is wrong, and §0 says why — inductor's actual materialisation is
1618.5 MiB; the measured figure is that number times hardware amplification. The
coincidence is a trap and is recorded here so nobody falls into it.


### 2.4 Every matmul in the lean step (job 907096)

303 matmul nodes, 735.4 MiB of operand bytes. The ones that matter for §5:

| B | M | K | N | count | ideal MiB | what |
|---|---|---|---|---|---|---|
| 1 | 6400 | 400 | 64 | 6 | 68.55 | `linear_dist`, pair_1 |
| 1 | 4800 | 400 | 64 | 6 | 51.56 | `linear_dist`, pair_2 |
| 1 | 6400 | 240 | 64 | 6 | 44.88 | `linear_pairvec`, pair_1 |
| 1 | 4800 | 240 | 64 | 6 | 33.75 | `linear_pairvec`, pair_2 |
| 800 | 1 | 32/64 | 24..64 | 6 each | 29–39 | the IPA contractions, as **batch-800 M=1 gemv** |
| **100** | **3** | **3** | **2560** | **6** | **35.18** | **`pair_vector_features`' `apply_inverse_t` (K=64)** |
| **100** | **3** | **3** | **1920** | **6** | **26.39** | **same, K=48** |
| 100 | 3 | 3 | 1280 | 6 | 17.60 | `direction_features`' `apply_inverse_t` (K=64) |
| 100 | 3 | 3 | 960 | 6 | 13.20 | same, K=48 |
| 6400/4800 | 3 | 3 | 3 | 6 each | 3.96 / 2.97 | `position_rotation_features`, as a **batch-6400 bmm of 3x3s** |
| 100 | 3 | 3 | 20 | 19 | 0.93 | `aa_frames_t`'s own frame-local coordinates |

The four bolded/adjacent rows are the same operation at two widths: **the
`(3,3)x(3,2560)` contraction contains the `(3,3)x(3,1280)` one**, because the
neighbour half of `pair_vector_features` is `direction_features` before its
normalisation. That is O1.

---

## 3. The exchange rate: what a MiB is worth

From §1.2, measured on this model, this GPU, at HEAD:

* **a Triton launch costs 1.06 us to exist** (N017's floor probe, reused)
* **a cuBLAS launch costs 5.98 us**, and 352 of them account for 2105 of the
  2166 us cuBLAS spends — so **GEMM-side bytes are free**
* Triton's 392 launches spend 1077.6 us, of which 415.5 us is launch floor;
  the remaining 662 us moves 2234 MiB of L2, i.e. **3.54 TB/s**, 54 % of this
  node's measured L2 rate. So on the Triton side bytes *are* time:
  **1 MiB of measured L2 ≈ 0.30 us; 1 MiB of logical traffic ≈ 0.62 us**
  (logical x 2.08 amplification).

The rule that follows, and every candidate below is scored against it:

> **a change pays iff `0.62 x (logical MiB removed) + 1.06 x (triton launches
> removed) + 5.98 x (cuBLAS launches removed) > 0`** — so 1 cuBLAS launch is
> worth 9.6 MiB of logical traffic, and 1 Triton launch 1.7 MiB.

---

## 4. Loop-invariant traffic: there is none left

The brief's highest-value pattern. Two independent checks, both negative.

**(a) Structurally duplicate subexpressions, post-CSE.** Scanning the *generated
code* (not the pre-CSE ATEN graph — the shipped `FastDenoiser` installs
`fx_graph_cse` as `joint_custom_post_pass`) for kernels launched more than once
with identical input arguments (`scripts/parse_code.py`, run over job 907101's
lean graph):

```
residual identical-argument launches after inductor CSE: 1 group, 4 redundant launches
  x5  triton_poi_fused_gelu_mul_scatter_add_where_zeros_like_83
```

That one kernel is a bare `tl.store(0.0)` zero-fill of two `(100, 512)` buffers —
the `torch.zeros_like(data)` that `index_sum` scatters into. 5 launches x
0.39 MiB = **1.95 MiB and 5 launches (~5 us, 0.16 %)**, and they are *not*
redundant: each block's scatter mutates its own buffer, so the five zero-fills
write five different destinations. **Nothing else in the forward is recomputed.**
P6 (the hoisted neighbour stages) plus N017/O1's CSE have closed this class
completely; the 700 duplicate groups / 2792 redundant nodes / 101.8 MiB visible
in the *pre*-CSE ATEN graph (job 907096) are exactly what the CSE pass deletes.

**(b) `pair_vector_features`' K-independent self half — N017's open lead.**
Confirmed present (the graph shows `x6 expand, 5.49 MiB, features.py:208`) and
**confirmed not worth removing as traffic**, because the traffic is not the
computation, it is the *GEMM operand*. The output must be materialised as
`(N, K, 240)` for `linear_pairvec` whatever the self half costs to produce.
Splitting the contraction into a self part `(N,120)@(120,64)` broadcast over K
and a neighbour part `(N,K,120)@(120,64)` removes ~112 MiB of logical traffic —
worth ~69 us by §3's rate — and adds **one cuBLAS launch per pair block, 12 in
all, +72 us**. Net ≈ 0, for a change that is a reassociation (so it could never
ship in `exact` and would need the full n=160 structural gate). **Rejected on
arithmetic, not measured** — the launch economics decide it before a job is
needed. This closes the lead.

What the self half *is* good for is something else entirely, and it is bit-exact:
see O1 in §5.

---

## 5. O1 — `pair_vector_features` shares the contraction that already exists

### 5.1 What it is

`pair_vector_features` and `direction_features` both push the gathered
neighbour atoms through `apply_inverse_t`, and ATen lowers both to `bmm`:

```
direction_features : bmm (100,3,3) x (100,3, K*A)     A=20  -> K=64: width 1280
pair_vector_features: bmm (100,3,3) x (100,3, K*2A)         -> K=64: width 2560
```

The 2560-wide one **contains** the 1280-wide one: its neighbour half is
`direction_features`' pre-normalisation `local`, bit for bit. Its *self* half is
`R_i^T (pos_i - ca_i)`, which does not depend on `k` at all and is exactly
`aa_frames_t(pos)[2]` — a tensor the block already holds. So the wide
contraction recomputes one thing the graph has and one thing it is about to
compute again, K times over.

O1 (`salad_torch/geometry/features.py`, flag `SALAD_OPT_PAIRVEC`) builds
`local` as `cat([local_self.expand(n,k,a,3), local_neigh])` instead. The
2560/1920-wide bmm disappears; inductor's CSE then shares the 1280/960-wide one
between the two call sites. Patch: `O1_pairvec_share.patch`.

### 5.2 Bit-exactness: REFUTED, and the way it fooled three checks

Three checks said yes before the fourth said no, and the order is the lesson:

1. **the function in isolation, CPU, 8 shapes** (48/64/100/200 residues x K=11/32/48/64,
   both thread counts): `torch.equal` **True**, max|d| = 0.0 at every shape.
2. **the compiled `forward_core`, CPU bitwise control, L=64** (job 907154, the
   same harness N017 used to certify its CSE pass): **bit_identical = True**,
   all eleven outputs.
3. **the full parity suite**, both arms (job 907188): **SUITE_EXIT=0**, four
   configs, geometry 38/38, stack, encoder, sampling — and O1's coordinate
   deviations are *better* than the baseline's on three of four configs
   (0.00001 vs 0.00002 A).
4. **the eager model on the parity fixtures, arm vs arm** (`scripts/x_eager_arms.py`,
   `o1_eager_arm_diff.txt`) — the check none of the above performs:

   | config | `pair_vector_features` calls that differ | `pos` max abs diff |
   |---|---|---|
   | default_vp | 10 / 16 | 9.06e-06 A |
   | default_vp_scaled | 10 / 16 | 7.63e-06 A |
   | default_ve_scaled | 10 / 16 | 1.14e-05 A |
   | **multimotif_vp** | **16 / 16** | **1.63e-04 A** |

   The feature tensor itself differs by 1–2 ulp (2.4e-07 on values of order 1).

**So O1 is a reassociation, not a rewrite of the same bits.** The width-20 GEMM
that produces `local_self` and the width-2560 one that used to produce it pick
different BLAS kernels, and whether the two agree is **data-dependent** — which
is why the synthetic shapes and the L=64 compiled control both came out exactly
0.0 and the real fixture data did not. *A single bitwise control on synthetic
inputs is not evidence of bit-exactness in this model.* Record that.

By the standard this repo already set — N017 rejected `TORCHINDUCTOR_FREEZING`
at **8.8e-06** precisely because it "would pass `test_model_parity` while
quietly not being fp32-exact", and N020 rejected the `mm` lowering at 4.2e-05 —
**O1 cannot ship in `exact`.** Its multimotif figure, 1.63e-04 A, is the largest
of the three and sits just under that config's 2.0e-04 A parity floor.

### 5.3 There is no bit-exact version of this optimisation

The two call sites contract at two different widths. Sharing forces one of them
to change width, so **whichever site is made to reuse the other changes its own
bits**: O1 moves `pair_vector_features`; the mirror-image transform (slice
`direction_features` out of the 2560-wide result, which is how N028's O6 is
written in its plan) moves `direction_features` instead, by the same mechanism
and the same magnitude. **N028's O6 is listed as "Bit-exact" in its plan; it is
not**, and this is the evidence. Either direction is a `fast`-preset candidate
needing N015's n=160 structural gate, never an `exact` one.


### 5.4 What it is worth (job 907152, one length per process, 40 iters)

| | `SALAD_OPT_PAIRVEC=0` (today) | `=1` (O1) | delta |
|---|---|---|---|
| lean step, L=100 | 3.379 ms | **3.131 ms** | **−7.3 %** |
| full step, L=100 | 4.174 ms | 3.921 ms | −6.1 % |
| 500-step design, L=100 | 1.69 s | 1.57 s | −7.1 % |
| lean step, L=200 | 4.586 ms | **4.244 ms** | **−7.5 %** |

**Kernel census, both arms, same job (L=100, lean, profiler):**

| | `=0` | `=1` (O1) | delta |
|---|---|---|---|
| GPU busy | 3326.9 us | 3084.2 us | **−242.7 us** |
| total kernels | 807 | 796 | **−11** |
| **cuBLAS** | 352 launches / **2163.0 us** | 341 / **1930.8 us** | **−11 launches, −232.2 us** |
| Triton | 392 / 1076.2 us | 392 / 1065.7 us | **0 launches**, −10.5 us |
| memcpy | 61 / 82.8 us | 61 / 82.9 us | 0 |

**Exactly the eleven cuBLAS launches predicted** — six from `pair_1`, six from
`pair_2`, plus one added back because `aa_frames_t(prev_pos)`'s frame-local
coordinates become live (CSE shares that one across all six blocks). **96 % of
the win is those eleven launches**, at 21.1 us each: well above N017's 5.98 us
launch floor, because a `(100,3,3) x (100,3,2560)` strided-batched sgemm is
also slow to *execute*. Triton's launch count does not move at all and its time
moves 1 %.

**−7.3 % is the largest single-change win recorded on this model since P7, and
it is not a traffic win.** The bytes it removes are ~61 MiB of GEMM operands,
worth essentially nothing by §3 (cuBLAS time is 97 % launch latency). What it
removes is **twelve `(100,3,3) x (100,3, K*2A)` cuBLAS launches** — the wide
`apply_inverse_t` bmm in every one of the twelve pair blocks. That is the
thesis of this whole item in one measurement: on this model you are paid for
launches, not for bytes, and a byte-motivated change only pays when it happens
to delete a launch.

### 5.5 Gates

| gate | result |
|---|---|
| `test_model_parity` x 4 configs (job 907188) | **PASS**; deviations 0.00001 / 0.00001 / 0.00002 / 0.00028 A vs the baseline arm's 0.00002 / 0.00002 / 0.00002 / 0.00014 |
| `test_geometry_parity` | **PASS 38/38** |
| `test_stack_parity`, `test_encoder_parity`, `test_sampling_parity` | **PASS** |
| **SUITE_EXIT** | **0**, both arms |
| CPU bitwise control, compiled `forward_core`, L=64 and L=100 (job 907154) | bit_identical **True** — *and it is not sufficient, see §5.2* |
| `test_design_quality.py --n 160 --preset fast`, O1 arm (job 907178) | **PASS, 0/160 diverged**, CA-CA 3.7628, Rg 13.01 (N015 reference: 3.7613–3.7628, 12.89, 0/160) |
| same, baseline arm (job 907217) | **PASS, 0/160 diverged**, CA-CA 3.7621, Rg 12.95 |

**The paired n=160 structural comparison (N015's method, same 160 seeds, both
arms, `scripts/cmp_quality.py` -> `o1_quality_n160.txt`):**

| metric | baseline (n=160) | O1 (n=160) | Welch | KS | Brown-Forsythe |
|---|---|---|---|---|---|
| CA-CA | 3.7621 ± 0.0138 | 3.7628 ± 0.0134 | p=0.661 | p=0.761 | p=0.770 |
| Rg | 12.950 ± 1.823 | 13.009 ± 2.053 | p=0.783 | p=0.915 | p=0.798 |
| diverged | **0/160** | **0/160** | | | |
| Rg > 18 A outliers | 4/160 | 6/160 | | | |

Nothing moves in mean, spread or shape. **O1 is validated for the `fast` preset
by the same standard N015 used to keep `steps_scale=0.5`** — and rejected from
`exact` by §5.2. It is not landed here: see §10.

---

## 6. How much of the 5.2x is irreducible

Answering the brief's fourth question directly, with the §0 decomposition:

| term | MiB | reducible | by what |
|---|---|---|---|
| distinct tensors the dataflow must materialise (N016's count) | 926.9 | **no** | — |
| the second touch of each of them (store **and** load) | +691.6 | **no** | identity: a kernel boundary is a round trip |
| — of which cross-kernel Triton→Triton intermediates | (164.8) | **yes** | better pointwise fusion; **~0.10 ms, 3 %** |
| — of which Triton→cuBLAS operands + cuBLAS→Triton results | (1269.6) | only with a fused-prologue/epilogue Triton matmul; not bit-exact (N020) | |
| SM↔L2 access amplification, x2.08 | +1753.6 | **partly** | layout. TF32 alone takes 382 MiB of it (§1.3); the `clone`-for-`bmm` class is 776 MiB (§2.2) |
| DRAM-side fill + writeback counted by `lts__t_bytes` | +1230.3 | **no** | it is the price of the 896.6 MiB that misses L2 |

**So: of the 4602 MiB, 1618.5 MiB is the dataflow's own arithmetic identity and
1230.3 MiB is the DRAM interface. 1753.6 MiB is access amplification, and of
that the single addressable block is the 776 MiB of `bmm` re-layout. The
classical fusion prize — the thing E7 was read as promising — is 164.8 MiB.**

Expressed as time at §3's rate, the whole reachable set is:

| item | logical MiB | launches | estimated ms | % of 3.393 ms | owner |
|---|---|---|---|---|---|
| einsum → mul+sum (kills the `clone` class) | −311.7 | −67 triton, but adds reductions | ~−0.20 to −0.30 | 6–9 % | N027 (O2) |
| remaining pointwise fusion (all 82.4 MiB of edges) | −164.8 | a handful | ~−0.10 | 3 % | inductor, not us |
| RBF fused into a Triton `linear_dist` prologue | −205 | −12 triton | ~−0.15 | 4 % | needs a custom matmul; not bit-exact |
| **O1 (§5)** | measured | measured | measured | | this item |

Nothing else in the forward is worth more than 1 %. **E7 should be closed**: the
memory-traffic gap is real but it is 4.9x of an inapplicable denominator, and
the part of it that any change can reach is worth single-digit percent, all of
it already inside N027's and N028's scope.

---

## 7. Portability

| finding | class | on a consumer card / sm_120 |
|---|---|---|
| the traffic decomposition (§0) | **PORTABLE** | the ratios are graph properties, not device properties; only the DRAM-side term grows as L2 shrinks |
| "bytes are free on the cuBLAS side" (§3) | **H100-DEPENDENT, and it gets WORSE elsewhere** | it holds because cuBLAS time here is 97 % launch latency. A consumer card has ~1 TB/s of L2 instead of 3.35, so the same 2.3 GiB of GEMM traffic costs ~3x more and could stop being free. Re-measure §1.2 before trusting the exchange rate off this node. |
| 1 MiB ≈ 0.30 us of Triton time | **H100-DEPENDENT** | scales with that card's L2 rate; on a 4090-class part expect ~1 us/MiB, i.e. **traffic reduction is worth ~3x more there** |
| TF32 cuts cuBLAS traffic 16 % (§1.3) | **PORTABLE** (TF32 exists on Ada and Blackwell) | worth more, for the same reason |
| the `clone`-for-`bmm` class (§2.2) | **PORTABLE** | it is a lowering property of `torch.einsum`, identical on any backend; removing it helps a small card more |
| L2 residency arguments | **H100-DEPENDENT** | this node has 50 MiB of L2 and an 80.4 % hit rate on a working set of a few MiB per stage. A 4090 has 72 MiB (more), an L40S 48 MiB, a 4070 36 MiB — the largest single live tensor is the 10.2 MiB `(6400,400)` distance feature, so the hit rate should hold down to ~32 MiB of L2 and degrade below that. |
| none of this item's analysis or the O1 patch uses TMA, wgmma, cluster launch, inline PTX, an arch guard, or any constant tuned to 132 SMs | | |

---

## 8. Sized leads, handed over

Everything below is measured here and belongs to a sibling item. The numbers are
the contribution of this item; none of it needs re-measuring.

**L1 — `torch.einsum` on 3x3 frames costs 79 cuBLAS launches (N028/O8).**
Of the 303 matmul nodes in the lean step, **79 have a `(...,3,3)` operand**:
67 are `apply_t`/`apply_inverse_t` (`(100,3,3)x(100,3,W)` for W in
{20, 64, 192, 960, 1280, 1920, 2560}) and 12 are `position_rotation_features`
(`bmm` over a **batch of 6400 3x3x3 products** — 345 kFLOP for a 5.98 us
launch). At N017's measured cuBLAS launch price that is **472 us, 14 % of the
3.393 ms step, for 107 MiB of operands and under 0.5 % of the arithmetic.**
Written as mul+sum they become pointwise+reduction nodes that inductor can fuse
into their consumers, so the upper bound is close to the whole 472 us. The catch
is the one N017 already hit on the IPA contractions: a 3-term reduction
reassociates, and that failed `test_sampling_parity` 7/64 there. It is the
largest single item in the forward and it is N028's.

**L2 — the `clone`-for-`bmm` class is 776 MiB and 67 launches (N028/O2).**
§2.2. Mechanism, from the generated code: `triton_poi_fused_clone_20` (the
single largest Triton kernel, 186.9 MiB, 63.1 us) gathers `k[neighbours]` with
the **neighbour index innermost**, because that is the layout `bmm` needs —
every consecutive thread reads a different residue's row, so the gather is
fully uncoalesced (9.8 MiB of L2 reads per launch to fetch 6.5 MB of distinct
data). mul+sum keeps the natural `(N, K, H, S)` layout, in which the channel is
innermost and the gather coalesces. So O2 removes a kernel, a cuBLAS launch
*and* the read amplification, three ways at once.

**L3 — the distance RBF is the only worthwhile prologue fusion.**
`linear_dist` reads a `(N, K, 400)` tensor that exists for one kernel's
lifetime: 9.77 MiB written and read back per pair block, **205 MiB logical and
12 Triton kernels** over the forward. A Triton matmul with the RBF in its
prologue removes all of it (~0.15 ms, 4 %). Plain `tl.dot`, so portable to
sm_120 — but N020 measured Triton GEMM templates as not bit-exact end to end,
so this can only ever be a `fast`-preset candidate behind an n=160 gate.

**L4 — dead ends, so nobody re-opens them.**
* `index_sum`'s `scatter_add` moves **zero** atomic traffic (§1.1). Not a target.
* The neighbour `sort` has the highest amplification in the forward (20.6x) and
  is still only 40.6 MiB (0.9 %). Its cost is time, not bytes.
* `pair_vector_features`' K-independent self half (§4b): rejected on launch
  economics, net ≈ 0.
* The measured L2 traffic (4576 MiB) being close to the *unfused* graph traffic
  (4145 MiB) is a coincidence of two unrelated quantities (§2.3), not evidence
  that fusion is not happening.

---

## 9. Method, and what the numbers are worth

**The four ncu traps in the brief were all live and all avoided**:
`TORCHINDUCTOR_COMPILE_THREADS=1`; `--nvtx-include "measure/"` with the trailing
slash; no grep over ncu's indented metric table (`scripts/parse_ncu2.py` parses
it); and every figure is `sum(Invocations x average)` — the unweighted sum of
this run's group averages is 522 MiB against the true 4576 MiB, an 8.8x
undercount. `parse_ncu2.py` additionally handles the `sector` unit, which
N016's parser silently dropped (its regex accepted only byte units).

**One new trap, found here.** ncu's summary prints sector counts with
thousands separators (`31,055.00`); a parser that does not strip them either
throws or, worse, matches a truncated number. And the sector metrics can print
in `sector`/`Ksector`/`Msector` — this run used only `sector`, but a longer run
would not.

**How firm each number is:**

| number | source | firmness |
|---|---|---|
| 4576.0 / 4602.4 MiB measured L2 | two ncu jobs | **hard**, ±0.3 % run-to-run |
| 3372.1 MiB SM-side, 0 atomic | ncu op_* sectors | **hard** |
| 926.9 MiB barrier minimum | traced ATEN graph | **hard** — reproduces N016 exactly |
| 735.0 MiB GEMM operands | traced graph shapes | **hard** |
| 883.4 MiB Triton logical | parsed `output_code` | **estimate, ±20 %**: a kernel that reads part of a buffer is over-counted, one that gathers is under-counted. The two biases oppose. |
| 82.4 MiB cross-kernel Triton→Triton | same parse | **estimate**, same caveat — but the verdict survives a 2x error either way |
| 1 MiB ≈ 0.30 us | 2234 MiB / (1077.6 − 415.5) us | **derived**, a fleet average over 392 kernels, not a per-kernel law |

**Why the logical figure is compared against the measured one at all.** Every
previous sizing of the fusion prize (N007's 0.37 ms roofline, N009's floor,
E7's 5.2x) compared a *measured* number against a *graph* number without
checking whether the two count the same events. §0's first two rows are that
check, and they are where the 5.2x goes.


**L5 — the per-step host-side input copies (unowned, and a correctness trap).**
`FastDenoiser._copy_in` writes all 19 `_CORE_ARGS` into the persistent buffers
on **every** step, but `dmap`, `omap`, `dmap_mask`, `flags`, `resi`, `chain`,
`batch`, `mask`, `aa_gt`, `pos_gt` and `cyclic_mask` do not change over a whole
run — only `pos_noised`, `prev_pos`, `prev_local`, `t_pos`, `t_seq`,
`seq_noised`, `condition` and `uniform` do. Skipping the unchanged ones is
bit-exact by construction (the buffer already holds those bytes) and removes
~11 host-launched `cudaMemcpyAsync` per step out of the 0.178 ms python wrapper
N016 measured. **Not implemented here, deliberately**: the only cheap guard is
`(id, data_ptr, _version, shape, dtype)` on the source tensor, and if it is ever
wrong the failure is a *silently stale conditioning map* — the P7 failure class.
Worth ~1 %; do it only with a design-quality gate, not a parity test.

---

## 10. What is landed, and what is handed over

**Landed: the records only.** No source file in the repo is changed by this
item.

**Handed to N028, with the gates already run.** `O1_pairvec_share.patch` is
this item's O1, and it is **the same transformation as N028's O6**
(`direction_features` and `pair_vector_features` share their contraction) —
N029 found it independently from the traffic attribution before reading N028's
plan. Rather than land it into a file N028 owns and create a textual conflict,
it is delivered as a patch with everything a landing needs:

* **−7.3 % at L=100, −7.5 % at L=200** (lean step, job 907152)
* **N028's plan calls O6 "Bit-exact". It is not** — up to **1.63e-04 A** on
  `multimotif_vp`, §5.2. The three checks that said otherwise, and why each was
  insufficient, are written out in §5.2; the mirror-image form of O6 in N028's
  plan has the same defect for the same reason (§5.3).
* full parity suite **SUITE_EXIT=0**, both arms (job 907188)
* `test_design_quality --n 160 --preset fast`: **0/160 diverged in both arms**,
  and the paired n=160 comparison finds nothing in mean, spread or shape
  (all p ≥ 0.66) — so it **is** validated for `fast` by N015's standard
* to land: flip `opt.PAIRVEC_SHARE`'s default to `False` and add a
  `pairvec_share` preset key set only in `fast`

**Handed to N028 as sized measurements, not patches:** L1 (79 cuBLAS launches
on 3x3 frame contractions, 472 us, 14 %), L2 (the `clone`-for-`bmm` class,
776 MiB / 67 launches, and the mechanism — `bmm`'s layout makes the neighbour
gather uncoalesced), L4 (four dead ends). §8.

**Handed to N030:** cuBLAS moves 2335 MiB of L2 in 2166 us, i.e. 1.13 TB/s =
17 % of the L2 rate, and 2105 of those 2166 us is the 5.98 us launch floor x
352. **Bytes are not why the GEMM side is slow**, so a GEMM change should be
scored on launches alone. TF32 does cut cuBLAS traffic 16 % (§1.3), which is
a portability argument for a small card, not a speed argument for this one.

**E7 should be marked closed** in N016's gap list, with the decomposition in
§0 as its resolution.
