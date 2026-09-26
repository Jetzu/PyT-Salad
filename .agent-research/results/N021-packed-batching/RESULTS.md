# N021 — packed-batch inference, and why it should not ship

**Verdict: DO NOT SHIP `--batch_size` as a speed knob.** It works, it is
correct, and it is faster — **+42 % / +16 % / +4.5 % / +3.6 % / 0 %** more
designs per second at L = 50 / 100 / 200 / 400 / 1000 on top of N019's
`--group_size` ceiling. But at n=160 per arm, replicated on two independent
seeds and with the obvious confound controlled, **packing measurably changes
the design distribution**, and the number that decides it is the amino-acid
composition: **alanine goes from 7.2 % of residues to 39.8 %, leucine from
46.8 % to 14.1 %, χ² = 13 984 on 19 dof, p = 0** between `--batch_size 1` and
`--batch_size 8`. The backbones follow: helix +0.06, strand −0.025,
coil −0.037, Rg +2.4 Å, every KS test p < 1e-4.

For scale, this project declined N018's `randhalf` lever over a helix/strand
shift of 0.019/0.019. This is three times larger, plus a sequence-composition
change that lever never had.

Jobs: 904375 (smoke), 904424 (parity), 904403-904420 (55-cell surface),
904421/904447 (quality arms, seeds 42 and 1234), 904436 (divergence),
904422 (`--batch_size 1` equivalence), 904430 (CPU semantics, `cpu` partition),
904581/904428 (roofline), 904586 (baseline control), 904595 (seed-set control),
904582 (ncu), 904623 (fp32 mechanism probe).

All numbers: H100 (132 SM, sm_90, 93.1 GiB, driver 610.57.04), torch
2.5.1+cu121, `default_vp-200k`, one length **and one (batch, group)** per
process, detached snapshot at `/hnvme/workspace/v181bb10-salad/n021tree`,
unmodified-HEAD control at `.../n021base`.

---

## 1. What was built

`--batch_size B` on `scripts/generate.py`, composable with `--group_size G`:
one captured replay advances **B × G** designs.

| file | what |
|---|---|
| `salad_torch/diffusion/sampling.py` | `pack_designs` / `split_packed_snapshot` — the inference-side counterpart of `data/collate.py`'s `concat_collate` |
| `salad_torch/modules/fast.py` | `batch_size` on `FastDenoiser` (every buffer sized for the flat axis) and on `GroupFastDenoiser`; `designs_per_replay`; `_autotune` divides by designs, not members |
| `scripts/generate.py` | `--batch_size`, `run_grouped` rewritten around phases |

salad has **no leading batch dimension**: a batch is several proteins
concatenated along one flat residue axis with a companion
`batch_index`/`chain_index`/`mask`, which is why `torch.stack` is the wrong
collate and `concat_collate` is the right one. `pack_designs` does exactly
that, **never pads** (`notes/04_PARITY_AND_PERF.md` B1 forbids it), places the
dense `(L, L)` pair maps **block-diagonally** rather than concatenating them,
and **renumbers `chain_index`** per member. Every field is packed by an
explicit rule; an unrecognised key raises rather than falling through.

`--batch_size 1` is today's path, and that is gated, not asserted: under
`SALAD_OPT_DETSEG=1`, `--batch_size 1 --group_size 4` writes PDBs **byte-identical**
to `--group_size 4` with no `--batch_size` at all, 4/4 designs (job 904422).

Two flag-handling details worth knowing: `--num_designs` not a multiple of
`--batch_size` runs the remainder through a second, smaller-B denoiser (one
extra compile, announced) rather than padding the pack; and a trailing partial
*group* re-captures at fewer members exactly as N019 did.

### 1.1 A bug found here, fixed by N022: the inductor FX-graph cache was off

N017/O1 installed its CSE pass as a closure inside `FastDenoiser.__init__`.
Inductor pickles `joint_custom_post_pass` into its FX-graph cache key, a local
function is unpicklable, and the failure is a **warning** ("Can't pickle local
object 'FastDenoiser.__init__.<locals>._cse_pass'"), after which the cache is
never written and every process recompiles from cold — measured **295 s per
process** at L=100, which is what made this sweep expensive.

Found independently here and by **N022**, which landed the fix (a module-level
`_cse_graph_pass`). N021's own duplicate was **reverted** so the two do not
collide; the measurements in this document were taken with the equivalent
module-level pass in place. Same function, same graph, same numbers.

---

## 2. Semantics: packing does NOT leak between designs

Full detail in `SEMANTICS.md` (job 904430, `cpu` partition, 48 cores — CPU on
purpose, because the CUDA forward's own run-to-run noise is 1.755e-02 Å, a
thousand times the effect being measured).

**The leak test.** With identical injected randomness, is design 0 sensitive to
*who else* is in the pack? `[0,1]` member 0 vs `[0,2]` member 0:

| L | 32 | 50 | 64 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|
| max Δ`pos` | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| Δ`aatype` | 0/32 | 0/50 | 0/64 | 0/100 | 0/200 | 0/400 |

Exactly zero at six lengths. Not "within a tolerance" — identical. And it is
not vacuous: at L=50 the `-1` sentinel is live and really does gather another
design's features into `blocks.1.pair_1.linear_dist` (2.675 apart between the
two packs), but the attention masks those columns to `-1e9`, the softmax
underflows to exactly 0, and the contamination dies there.

**What does differ from a standalone run** is 1e-4 to 5e-2 Å per forward, and
the module-level bisect names it: the first divergence is a linear layer whose
inputs are bit-identical (`blocks.0.update.mlp.layers.0`, 3.8e-06). **A GEMM
with the same operands and a larger `M` picks a different tiling and a
different accumulation order.** Packing reassociates the matmul reductions;
it does not change the computation. Downstream that flips near-ties in the
stable argsort, so neighbour sets move — but late, lightly, and **never at a
`-1` slot**, so B1 is not the mechanism.

**One real leak was found and fixed inside `pack_designs`:** concatenating
`chain_index` as-is (which is what upstream's own training collate does) fuses
the pack in `Update`'s per-chain global gate and in the relative-position
feature's chain length. Measured before the fix: `local` off by 34-38, `pos`
by ~4 Å, and it moved the last member too. Recorded, out of scope: packed
**training** batches in salad share chain ids and take both leaks.

**Over a full rollout** it is a different design anyway (job 904436, DETSEG=1,
identical injected randomness, 250 steps):

| | max Δ`pos` | Δ`aatype` |
|---|---|---|
| within-arm floor (B=1 twice) | **0.000000e+00** | **0/100**, **0/200** |
| B=4 vs B=1, L=100 | 35.4 – 67.4 Å | 59 – 88 / 100 |
| B=4 vs B=1, L=200 | 54.9 – 71.4 Å | 161 – 187 / 200 |

The floor being exactly zero is what makes this arm discriminating at all.

---

## 3. The trade-off surface — 55 cells, zero failures

`surface_table.txt` has all of it. Best cell per length, against the
`B=1, G=1` baseline **and** against N019's group-only ceiling:

| L | baseline | best group-only (N019) | best packed | packing's gain |
|---|---|---|---|---|
| 50 | 405.5 d/s | G=16 → 975.0 (2.40×) | **B=8 G=4 → 1384.5 (3.41×)** | **+42.0 %** |
| 100 | 343.9 | G=16 → 613.4 (1.78×) | **B=8 G=2 → 712.7 (2.07×)** | **+16.2 %** |
| 200 | 247.1 | G=8 → 340.5 (1.38×) | **B=4 G=2 → 355.7 (1.44×)** | **+4.5 %** |
| 400 | 147.8 | G=4 → 171.1 (1.16×) | **B=2 G=2 → 177.2 (1.20×)** | **+3.6 %** |
| 1000 | 68.1 | G=2 → 70.1 (1.03×) | B=1 G=2 → 70.1 (1.03×) | **0 %** |

At L=1000 packing is a **loss**: B=2 is 0.93× and B=4 is 0.84× the baseline.

**The knee is a flat-axis length, not a batch size.** Every winning cell has
`B × L` between 400 and 800 residues: L=50→B=8 (400), L=100→B=8 (800),
L=200→B=4 (800), L=400→B=2 (800), L=1000→B=1. That is the recipe, and it is
a consequence of the arithmetic below.

**Peak VRAM** (reserved, the whole process) at the best cell: 6.5 GiB at L=50
(32 designs, 202 MiB/design), 6.6 GiB at L=100 (16 designs, 413 MiB/design),
6.6 GiB at L=200 (8 designs, 826 MiB/design), 6.6 GiB at L=400 (4 designs,
1.65 GiB/design), 8.4 GiB at L=1000 (2 designs, 4.2 GiB/design). **VRAM per
design falls with B at short L and rises with B at long L** — at L=1000,
B=1→4 takes it from 5966 to 6845 MiB/design, because the B² pair tensors
overtake the per-design activations. On a 24 GB card every recommended cell
fits; the binding constraint there is the same one N019 found, and the
existing `memory_bound_group` measurement still governs the group axis.

**Where the bottleneck is:** still launch/serialisation-bound, at every cell.
Nothing in this surface is memory-bound or arithmetic-bound (§4).

---

## 4. How far does it get? The negative result

N016's rooflines on this node: 66.91 TFLOP/s fp32 FMA, 6.53 TB/s L2,
2.94 TB/s DRAM, 10.014 GFLOP of matmul per lean L=100 forward.

At L=100, all at 8 designs per replay (jobs 904428, 904581):

| config | replay (lean) | executed | % fp32 peak |
|---|---|---|---|
| B=1 G=1 (one design) | 2.699 ms | 3.710 TFLOP/s | 5.55 % |
| B=1 G=8 | 11.031 ms | 7.262 TFLOP/s | 10.85 % |
| **B=2 G=4** | 10.700 ms | **7.487 TFLOP/s** | **11.19 %** |
| B=4 G=2 | 11.427 ms | 7.011 TFLOP/s | 10.48 % |
| B=8 G=1 | 11.765 ms | 6.810 TFLOP/s | 10.18 % |

**Grouping already took arithmetic from 5.6 % to 10.9 % of peak. Packing adds
0.3 points, and past B=2 it goes backwards.** At the best throughput cell
(B=8 G=2, 16 designs in 21.774 ms) it is 7.36 TFLOP/s = **11.0 % of fp32
peak** — the same 11 % as everything else. At L=50 the picture is the same
(B=1 G=1 3.33 %, B=1 G=16 9.65 %).

**This is the answer to "does batching convert the idle 95 % of arithmetic
into throughput?" — no. It converts idle *launch slots*, exactly as grouping
does, and the arithmetic ceiling of this dataflow on this card is ~11 % of
fp32 peak regardless of how the designs are arranged.**

And the reason is measurable, not speculative: `flop_overhead_x = 1.0000` at
every batch size. **The matmul FLOPs scale exactly linearly with B.** The
superlinear cost of packing is therefore entirely in the *non-matmul* work —
salad's neighbour selection is a global stable argsort over the flat residue
axis, its Gumbel field is `(draws, N, N)` drawn fresh every step, and
`prepare_condition` materialises `(N, N)` and `(N, N, 9)` maps. All three grow
as B² while the designs grow as B. That is why the knee sits at a fixed flat
length and why L=1000 loses outright.

A block-diagonal-aware neighbour selection and Gumbel draw would remove the B²
term and is the only thing that would make packing scale — but it lives in
`geometry/neighbours.py` and `modules/diffusion.py`, changes the arithmetic,
and would break parity. Not attempted.

---

## 5. The quality gate — and the controls that make it mean something

n=160 per arm at L=100, `fast` preset, N015's methodology and script, **twice**
(seed 42 job 904421, seed 1234 job 904447). Arms differ only in `--batch_size`;
`--group_size` is chosen so every arm advances 8 designs per replay, and
grouping is bit-identical (N019) so it cannot be a confound.

Converged designs only (diverged dropped), seed 42 / seed 1234:

| arm | n | CA-CA (Å) | Rg (Å) | helix | strand | coil | diverged |
|---|---|---|---|---|---|---|---|
| a_b1 (B=1) | 134/135 | 3.875 / 3.861 | 21.14 / 21.29 | 0.890 / 0.878 | 0.029 / 0.035 | 0.081 / 0.087 | 26 / 25 |
| b_b2 (B=2) | 142/141 | 3.890 / 3.886 | 22.21 / 22.07 | 0.918 / 0.914 | 0.016 / 0.021 | 0.066 / 0.065 | 18 / 19 |
| c_b4 (B=4) | 150/147 | 3.912 / 3.905 | 23.48 / 23.03 | 0.947 / 0.936 | 0.003 / 0.015 | 0.050 / 0.049 | 10 / 13 |
| d_b8 (B=8) | 152/155 | 3.915 / 3.911 | 23.66 / 23.66 | 0.947 / 0.943 | 0.006 / 0.008 | 0.044 / 0.051 | 8 / 5 |

Monotone in B, reproduced on both seeds. Welch vs a_b1 at B=8: CA-CA
p=0.000 (both seeds), Rg p=0.000, helix p=0.000, coil p=0.000 — these survive
Bonferroni (α=0.0028). Two-sample KS, the shape test: **every metric
p < 1e-4** at B=4 and B=8. Fisher on diverged: p=0.0017 / p=0.0002.

**Amino-acid composition**, the sharpest population-level signal (job 904447):

| | a_b1 | b_b2 | c_b4 | d_b8 |
|---|---|---|---|---|
| ALA | 0.0716 | 0.2441 | 0.3342 | **0.3979** |
| LEU | 0.4679 | 0.2866 | 0.1976 | **0.1406** |
| THR | 0.0074 | 0.0828 | 0.1145 | 0.1340 |
| LYS | 0.0573 | 0.1201 | 0.1481 | 0.1606 |
| GLY | 0.0846 | 0.0537 | 0.0441 | 0.0146 |

χ² vs a_b1: 5 092 (B=2), 9 846 (B=4), **13 984 (B=8)**, all on 19 dof, p=0.
The packed sampler drifts toward poly-alanine helices.

### 5.1 The confound, and the control that kills it

`design_generator` keys a design's RNG stream on its global index and the
grouped path keys a *pack* on its first design — so arm a_b1 used design
indices 0..159, b_b2 used 0,2,4,…, c_b4 0,4,8,… and d_b8 0,8,16,…. **The four
arms sampled four different sets of seeds**, and the whole trend could have
been a property of the seed sets.

Job 904595 removes it: generate **1280 designs at `--batch_size 1`**, then
subsample strides 1, 2, 4, 8 — four **unpacked** arms on exactly the seed sets
the four packed arms used.

| arm (converged) | n | CA-CA (Å) | Rg (Å) | helix | strand | diverged |
|---|---|---|---|---|---|---|
| s1 (0,1,2,…) | 137 | 3.8738 | 20.90 | 0.897 | 0.015 | 23 |
| s2 (0,2,4,…) | 138 | 3.8797 | 21.44 | 0.897 | 0.014 | 22 |
| s4 (0,4,8,…) | 137 | 3.8788 | 21.03 | 0.897 | 0.023 | 23 |
| s8 (0,8,16,…) | 135 | 3.8750 | 20.87 | 0.895 | 0.021 | 25 |

**Flat.** Largest helix difference 0.002; every Welch p ≥ 0.10. The seed sets
explain nothing, so **the trend across the packed arms is packing.**

### 5.2 A pre-existing regression on this branch, reproduced on unmodified HEAD

Job 904586 ran three arms to check whether the *baseline* was sane, because
a_b1's 26/160 diverged and helix 0.890 do not match N015's recorded `fast`
numbers (0/160 diverged, helix 0.580, Rg 12.90, measured at commit c5ab7a0):

| arm | tree | path | helix (conv.) | diverged |
|---|---|---|---|---|
| L1 | **unmodified HEAD fd9ba32** | legacy `--fast` | 0.850 | **41/160** |
| L2 | unmodified HEAD | `--group_size 8` | 0.897 | 17/160 |
| L3 | N021 worktree | legacy `--fast` | 0.873 | 26/160 |

**Unmodified HEAD's plain `--fast` path already diverges on 26 % of designs at
L=100 and already sits at helix ~0.86.** Something between c5ab7a0 and
fd9ba32 moved this distribution; it is not N021's and it is not `--batch_size`.
It is flagged here and nothing more — I did not bisect it. Note also that the
diverged *count* is a noisy statistic: L1 and L3 are the same computation and
gave 41 vs 26. The converged-only means are stable (0.850 vs 0.873).

L2 vs L1 does suggest grouping alone shifts helix by ~0.047, in the same
direction as packing, which the stride control does not address (it only
varies strides *within* the independent-stream family, not one-stream vs
many-streams). **Unresolved**, ~0.03-0.05 helix, and it belongs to N019's RNG
scheme rather than to this task.

### 5.3 The standard suite — unchanged

Job 904424, on the N021 worktree:

| test | result |
|---|---|
| `test_model_parity` `default_vp` | max **0.00002 Å** |
| `test_model_parity` `default_vp_scaled` | max **0.00002 Å** |
| `test_model_parity` `default_ve_scaled` | max **0.00002 Å** |
| `test_model_parity` `multimotif_vp` | max **0.00020 Å** |
| `test_geometry_parity` | **38/38** |
| `test_stack_parity` / `test_encoder_parity` / `test_sampling_parity` | OK / OK / OK |
| | **`SUITE_EXIT=0`** |

---

## 6. Recommendation

**Do not ship `--batch_size` as a recommended or default path.** The deciding
number is the composition χ² of 13 984 (ALA 0.072 → 0.398), backed by a
Bonferroni-surviving, twice-replicated, confound-controlled shift in every
structural metric — bought for +16 % throughput at L=100 and **+4 % at the
lengths above 200 where the extra speed would actually matter**. `--group_size`
exists precisely because it is semantically free; this is not, and it does not
raise the arithmetic ceiling either (§4). The one escape hatch — "it is just
TF32" — is closed: the fp32 replicate (§5.4) reproduces the shift and makes it
larger.

The code is correct and gated and is worth keeping in the tree as an
explicitly experimental flag — it already prints its own warning on every run,
`--batch_size 1` is byte-identical to today, and the whole parity suite is
untouched. It must not enter `--group_size auto`, any preset, or any default.

### 5.4 It is NOT a TF32 artifact — the fp32 replicate reproduces it

The obvious escape hatch was that all three runs above used `--preset fast`,
i.e. TF32. Job 904623 repeats the four arms at `--preset exact` (fp32, full
500/400 schedule), n=160 each, converged only:

| arm | n | CA-CA (Å) | Rg (Å) | helix | strand | coil | diverged |
|---|---|---|---|---|---|---|---|
| a_b1 (B=1) | 118 | 3.8741 | 21.91 | 0.904 | 0.010 | 0.086 | 42 |
| b_b2 (B=2) | 129 | 3.9066 | 24.88 | 0.947 | 0.003 | 0.050 | 31 |
| c_b4 (B=4) | 144 | 3.9119 | 25.31 | 0.957 | 0.003 | 0.040 | 16 |
| d_b8 (B=8) | 151 | 3.9196 | 25.52 | **0.967** | 0.001 | 0.032 | **9** |

**Same direction, same size, and if anything stronger**: helix +0.063
(t = +12.6, p = 0.000), Rg +3.60 Å (t = +9.9), coil −0.054 (t = −11.6), every
Welch p = 0.000 at B=8. Composition: ALA 0.0739 → **0.4154**, χ² = 10 248
already at B=4. So the shift is a property of **packing**, not of TF32, and the
mechanism has to be the fp32 GEMM reassociation that §2 identified — larger
`M`, different accumulation order, systematically fewer numerical blow-ups and
a rollout that lands more often in the model's dominant all-helix mode.

(Note `exact` at B=1 diverges on 42/160 — worse than `fast`'s 26/160 — so the
§5.2 HEAD regression is preset-independent too.)

**Could not measure / left open:**
* the quality gate is L=100 only. L=200 and L=400, where the speed gain is 4 %,
  were not tested at n=160.
* the mechanism behind fewer diverged designs at larger B. The obvious
  hypothesis — larger-M GEMMs accumulate more stably, so a numerical blow-up
  fires less often — is untested.
* the pre-existing HEAD regression in §5.2 was reproduced, not diagnosed.
* Blackwell: torch 2.5.1+cu121 cannot target sm_120, so every portability
  claim here is an argument from the primitives (`torch.cuda.Stream`,
  `CUDAGraph`, `torch.compile`), not a measurement.
