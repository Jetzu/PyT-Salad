# N021 §A — what packed batching actually changes (CPU, deterministic)

Job 904430, `cpu` partition, 48 cores. **CPU on purpose and not as a fallback**:
the question is whether packing changes the arithmetic, and the CUDA forward's
own run-to-run noise is 1.755e-02 A (N009) — a thousand times the effect being
measured. CPU is bit-deterministic, so `0.000000e+00` here is a statement and
not a coincidence.

Scripts: `x_packmate_cpu.py` (the leak test), `x_neighbourdiff_cpu.py` (do the
neighbour SETS move?), `x_hookdiff_cpu.py` (module-level first-divergence
bisect). All three use identical INJECTED randomness per design — one `(L, L)`
Gumbel field per design placed on the diagonal block of the packed
`(B*L, B*L)` field, one noise tensor concatenated — so the only free variable
left is a design's offset in the flat residue axis.

## A.1 The leak test: is design 0 sensitive to WHO ELSE is in the pack?

| L | `[0,1]` member 0 vs `[0,2]` member 0 | |
|---|---|---|
| 32 | `pos` **0.000000e+00**, `local` **0.0**, `aatype` 0/32 | |
| 50 | `pos` **0.000000e+00**, `local` **0.0**, `aatype` 0/50 | |
| 64 | `pos` **0.000000e+00**, `local` **0.0**, `aatype` 0/64 | |
| 100 | `pos` **0.000000e+00**, `local` **0.0**, `aatype` 0/100 | |
| 200 | `pos` **0.000000e+00**, `local` **0.0**, `aatype` 0/200 | |
| 400 | `pos` **0.000000e+00**, `local` **0.0**, `aatype` 0/400 | |

**Swapping a pack-mate for a completely different protein changes design 0 by
exactly nothing, at six lengths.** No cross-design information reaches a
packed design's output. That is the property that matters and it is the
strongest form the test has: not "below a tolerance", identical.

This is *not* vacuous. The intermediate tensors ARE contaminated: at L=50 the
`-1` no-neighbour sentinel is live (the three stages want 16+16+32 = 64
neighbours out of 49 candidates), `-1` wraps to the last row of the FLAT
array, and `blocks.1.pair_1.linear_dist` differs by **2.675** between
`[0,1]` and `[0,2]` — i.e. another design's pair features really are gathered
into the tensor. `SparseInvariantPointAttention` masks those columns to
`-1e9` before the softmax, which underflows to exactly 0, and the second
post-softmax mask zeroes them again. The contamination is destroyed exactly,
and the end-to-end bit-identity above is the proof.

## A.2 So what DOES differ from a standalone run?

| L | alone vs member 0 of a pack of 2 | alone vs member 1 of `[1,0]` (design 0 LAST) |
|---|---|---|
| 32 | 1.58e-04 A | 4.67e-05 A |
| 50 | 2.74e-04 A | 2.19e-04 A |
| 64 | 6.63e-05 A | 4.86e-05 A |
| 100 | 1.92e-04 A | 1.75e-04 A |
| 200 | 1.64e-03 A | 1.22e-03 A |
| 400 | 5.17e-02 A | 2.08e-04 A |

Per ONE forward, with identical randomness. `aatype` 0 differences everywhere.

The module-level bisect says what this is. At L=50, walking the 301 leaf-module
outputs in order, the first differences are ~1e-6 relative and they appear in
**linear layers whose inputs are bit-identical** — e.g. `blocks.0.update.mlp.layers.0`
differs by 3.8e-06 while everything feeding it is 0.0. A GEMM with the same
operands and a different `M` (100 rows instead of 200) picks a different
tiling and a different accumulation order. **Packing reassociates the matmul
reductions; it does not change the computation.** The same class of change as
N014's batched condition-branch GEMM, which `exact` already ships.

Downstream, that 1e-6 flips near-ties in the stable argsort, so the neighbour
SETS do move — but late and lightly, and never at a `-1` slot:

| L | neighbour calls whose selection differs (of 27) | entries | `-1` present? |
|---|---|---|---|
| 50 | 0/27 | — | — |
| 100 | 1/27 (call 23, block 5) | 2 | `-1` alone=0 packed=0 |
| 200 | 1/27 (call 26) | 8 | `-1` alone=0 packed=0 |
| 400 | 7/27 (calls 20-25) | 2-97 | `-1` alone=0 packed=0 |

## A.3 The B1 hazard the brief warned about does NOT fire here

`notes/04_PARITY_AND_PERF.md` B1 measured 2.34-4.94 A between a protein run
alone and the same protein batched or padded. Two things make that not happen
at production lengths in this implementation:

1. **No padding.** `pack_designs` concatenates exact-length designs. B1's
   4.94 A case was "L=6 alone vs L=6 + 4 masked padding residues".
2. **The `-1` write is defused across members.** B1's mechanism is that the
   stage-chaining `distance[i, neighbours] = update` writes the *original
   finite* distance back at column N-1 for every `-1` slot, un-masking residue
   N-1 so it is selected twice. In a pack, column N-1 belongs to the LAST
   member, so for every other member that entry is already `+inf` (killed by
   `same_item`) and the write puts `+inf` back — harmless. Only the last
   member behaves as it would alone. Every neighbour call that actually moved
   (A.2) had **zero** `-1` entries in either arm, so B1 is not the mechanism
   for any of the residual.

## A.4 One real leak found and fixed inside `pack_designs`

The first version concatenated `chain_index` as-is (which is what upstream's
own training collate does — `data/dataset.py` step 8 renumbers
`_numerical_chain_index` *local to each part*). Two of salad's reductions key
on `chain_index` ALONE, not on `batch_index`:

* `modules/blocks.py` `Update`'s per-chain global gate,
  `index_mean(chain_gate * local_update, chain, mask)`, in every block;
* `geometry/features.py`'s relative-position feature, whose chain length is
  `index_count(chain, ones)`.

With every member calling its first chain 0, both fuse the whole pack.
Measured at L=100, B=2, one forward: `local` off by 34-38, `pos` by ~4 A — and
it moved the LAST member too, so it was never B1. `pack_designs` now offsets
each member's chain ids past the previous member's maximum; only
chain-relative comparisons are ever made, so the offset is invisible within a
member, and the table in A.1 is the result.

**Recorded, out of scope, not fixed:** packed *training* batches in salad
share chain ids across items and therefore take both of these leaks.
