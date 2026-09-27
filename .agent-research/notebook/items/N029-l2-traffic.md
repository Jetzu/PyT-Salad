# N029 · The 5.2x L2 memory-traffic gap: attribution and reduction

type: research · date: 2026-09-27 · status: done

## Goal

Attribute the 4.81 GiB of L2 traffic N016 measured for one lean forward against
its 926.9 MiB graph-derived minimum (gap row E7), find the loop-invariant
traffic, reduce what is cheap and safe, and say honestly how much of the 5.2x
is irreducible.

## Plan

Measurement-led, three independent views of the same forward at HEAD 98bc61b,
L=100, preset `exact`:
1. Nsight Compute per-kernel, `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`, with the
   read/write/atomic/reduce sector split (N016 had only the total).
2. Inductor's generated `output_code` parsed into per-buffer and
   per-producer/consumer LOGICAL traffic - the number the measured one should
   be compared against.
3. The traced ATEN graph for the unfused upper bound, the barrier minimum, the
   matmul shape census and a structural duplicate scan.
Score every candidate against the measured launch/byte exchange rate, not
against bytes alone.

## Changes

`salad_torch/geometry/features.py` (O1, behind `SALAD_OPT_PAIRVEC`, default on)
+ the flag in `salad_torch/opt.py`.

## Evidence

See `RESULTS.md`. Jobs 907096 (graph+baseline), 907097/907131 (ncu), 907101
(output_code), 907140 (kernel-time census), 907141 (`fast`/TF32 ncu),
907152 (O1 A/B), 907154 (CPU bitwise control).

## Outcome

**E7 closes as a negative result.** Measured at HEAD: 4576-4602 MiB of L2 per
lean forward (N016's 4814.85 reproduces; the drop is P7 being off). The 5.2x
over N016's 926.9 MiB "minimum" decomposes as **1.75x x 2.08x x 1.36x**:

* **1.75x is definitional** - N016's minimum counts each barrier tensor ONCE,
  and any kernel-level implementation stores it and loads it. The real floor is
  926.9-1853.8 MiB and inductor's actual logical traffic is **1618.5 MiB**,
  inside that band. There is no 3.9 GiB of un-fused intermediates.
* **2.08x is SM-side access amplification** (32-byte sectors, GEMM tile
  re-reads, gathers). cuBLAS 3.18x, Triton 2.53x.
* **1.36x is DRAM-side fill/writeback** that `lts__t_bytes` also counts.
  Atomic traffic is exactly 0.00 - the `index_sum` scatter_add is not a target.

**The pointwise-fusion prize is 82.4 MiB of cross-kernel Triton->Triton reads
(~0.10 ms, 3 %), not 3.9 GiB.** The one large addressable class is layout, not
fusion: **776.3 MiB (17.0 % of all L2 traffic) and 67 launches are `clone`
kernels that only re-lay-out a gather for `extern_kernels.bmm`**, and `bmm`'s
layout is also what makes that gather uncoalesced. **Loop-invariant traffic: no
residue** - after P6 + N017's CSE the only repeated launch in the whole forward
is a 0.39 MiB zero-fill x5.

**Exchange rate, measured**: cuBLAS spends 2105 of its 2166 us on the 5.98 us
launch floor, so GEMM bytes are free; Triton runs at 3.54 TB/s net of its own
floor, so 1 MiB of logical traffic = 0.62 us and 1 cuBLAS launch = 9.6 MiB.

**O1 (delivered as a patch, NOT landed):** `pair_vector_features` reuses
`aa_frames_t`'s frame-local coordinates and `direction_features`' contraction,
deleting twelve wide `(100,3,3)x(100,3,K*2A)` bmms. **-7.3 % at L=100 / -7.5 %
at L=200** - the largest single-change win since P7, and a *launch* win, not a
traffic win, which is the item's thesis in one measurement: the two-arm census
shows **-11 cuBLAS launches and -232.2 us of the -242.7 us total** (21.1 us
each - above the 5.98 us floor, because a `(100,3,3)x(100,3,2560)` strided
batched sgemm is also slow to execute), with the Triton launch count
**unchanged at 392** and Triton time moving 1 %. Parity suite
SUITE_EXIT=0 both arms; `--n 160 --preset fast` 0/160 diverged both arms with
a paired comparison finding nothing (all p >= 0.66). **But NOT bit-exact**
(1.63e-04 A on multimotif_vp) - three checks said it was, including the CPU
bitwise control N017 used, because the agreement is DATA-dependent. So it is a
`fast`-only candidate. **It is the same change as N028's O6, which that plan
calls "Bit-exact"; it is not.**

Corrects N016: its 926.9 MiB is a distinct-tensor count, not a traffic floor.

## Outputs

`.agent-research/results/N029-l2-traffic/`
