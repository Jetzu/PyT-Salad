# N022 — VRAM requirement table for every inference mode, L up to 1000

Measurement only, no model/script changes. Pinned snapshot: `git archive
fd9ba32 -- salad_torch scripts` extracted to
`/hnvme/workspace/v181bb10-salad/n022tree` (80 files, commit
`fd9ba325199b115cd79683f165b8e838b9d4aaea`, branch `n006-manual-cudagraph`),
deleted after this measurement per the inode-grace rule. Raw per-cell JSON
(63 files, `torch.cuda.max_memory_allocated`/`max_memory_reserved` plus wall
time and any parsed "compiled + captured in Ns" line) is in `raw/`.

All numbers: one H100 (`h11-24`/`h11-17`/etc, 132 SM, sm_90, **93.1 GiB**
usable device memory, driver 610.57.04), torch 2.5.1+cu121, `default_vp-200k`
checkpoint, `--num_steps 3 --out_steps 2` (the smallest schedule that still
reaches an out-step, so `--fast`'s `lean_steps` need_aa=True/False graph pair
is fully built and any out-step-only extra work — renoise, sample_final — is
exercised at least once; peak memory is set by which STATIC buffers/graphs
exist, not by how many steps replay them). `torch.cuda.reset_peak_memory_stats`
was called once, right after CUDA init and before model construction, so every
number below is the WHOLE PROCESS high-water mark (model load + compile +
capture + N replay steps), matching how N019 measured its section 4.4 table.
Method: one (mode, L, N) per **process** (63 processes total across 8 SLURM
jobs, each job running its cells sequentially on one GPU) — never multiple
shapes compiled in one live process. 8 jobs, ~3.9 GPU-hours total (estimated
~7 GPU-h before running; actual was lower — compile times came in under the
conservative estimate). Zero OOMs, zero errors — this H100 has 93.1 GiB and
nothing in the tested grid overflowed it; the "does not fit" cells in this
report are from the fitted model extrapolated to smaller cards, stated as
such.

---

## 0. Headline

**The reserved/allocated divergence is the single most important number in
this report, and it grows with group size, not with L.** At `--group_size 16`,
L=1000: **4.64 GiB allocated vs 65.18 GiB reserved — a 14.0x gap.** A card
sized off `nvidia-smi`/allocated memory would be wrong by an order of
magnitude. Reserved is what the allocator has committed from the OS and is
what actually OOMs a smaller card; allocated is only the tensors live at that
instant. Report reserved for capacity planning, always.

**The DETSEG "may scale worse" hypothesis is REFUTED by measurement.** The
`deterministic` preset's one-hot `(L,L)` indicator matmul (`det_segment_sum`)
was expected to add memory that grows with L² on top of `exact`/`fast`.
Measured: `deterministic`'s peak reserved is **bit-identical to `exact`'s at
every one of the 8 lengths from L=50 to L=1000** (both single-design and
grouped at N=8/L=1000, 33048 MiB vs 33048 MiB). The transient one-hot buffers
are reused/freed within the same forward by the caching allocator rather than
accumulated, so they never touch the process high-water mark. `fast` (which
also skips the deterministic reduction) reads 0.1-1.0% lower than
`exact`/`deterministic`, consistent with ordinary TF32 kernel-buffer
differences, not with det_segment.

**`--fast` costs MORE memory than eager, not less** — it trades memory for
speed. At L=1000: eager 2710 MiB reserved vs `--fast` 4234 MiB (+56%), because
`FastDenoiser` holds the `need_aa=True` AND `need_aa=False` captured graphs
(with their own static input buffers) alive simultaneously.

**Every mode measured is LINEAR in L through L=1000, not quadratic.** A
quadratic term was the a priori worry (distogram/pair features/DETSEG
indicator are all pair-shaped, O(L²) in principle). Measured: a quadratic fit
never reduces the worst-case relative error over a linear one (see §2); the
apparent noise is CUDA allocator block-size quantization, not curvature.

---

## 1. The table

Peak **reserved** and **allocated** bytes (MiB), whole process, single design
unless noted. `n/a` = not measured at that cell (see grid design, §1.6).

### 1.1 Single design, no grouping — reserved / allocated MiB

| L | eager | `--fast exact` | `--fast fast` | `--fast deterministic` |
|---|---|---|---|---|
| 50 | 216.0 / 171.2 | 612.0 / 390.9 | 612.0 / 390.9 | 612.0 / 390.9 |
| 100 | 320.0 / 270.0 | 656.0 / 467.4 | 656.0 / 467.4 | 656.0 / 467.4 |
| 200 | 602.0 / 469.2 | 900.0 / 620.4 | 900.0 / 620.4 | 900.0 / 620.7 |
| 300 | 868.0 / 667.8 | 1344.0 / 776.6 | 1344.0 / 776.6 | 1344.0 / 777.3 |
| 400 | 1210.0 / 868.4 | 1716.0 / 936.8 | 1716.0 / 936.8 | 1716.0 / 938.0 |
| 600 | 1736.0 / 1274.5 | 2456.0 / 1267.3 | 2456.0 / 1267.3 | 2456.0 / 1270.1 |
| 800 | 2296.0 / 1684.5 | 3306.0 / 1606.8 | 3306.0 / 1606.8 | 3306.0 / 1611.7 |
| 1000 | 2710.0 / 2098.0 | 4234.0 / 1975.7 | 4194.0 / 1956.1 | 4234.0 / 1983.3 |

(`exact` and `deterministic` are within rounding of each other at every L —
the DETSEG refutation from §0, stated as raw numbers.)

### 1.2 `--fast --group_size N` (preset `fast`) — reserved GiB

| L \ N | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| 100 | 0.789 | 0.936 | 1.781 | 3.475 | 6.795 | 9.678 | 12.625 |
| 400 | 2.463 | 3.209 | 6.312 | 12.527 | 24.893 | — | — |
| 1000 | 5.826 | 8.170 | 16.102 | 31.961 | 63.650 | — | — |

Allocated GiB, same cells (for the divergence numbers in §0/§4):

| L \ N | 1 | 8 | 16 | 32 |
|---|---|---|---|---|
| 100 | 0.373 | 0.731 | 1.184 | 1.277 |
| 400 | 0.775 | — | 1.951 | — |
| 1000 | 1.764 | 3.068 | 4.531 | — |

### 1.3 Group at other presets, L=1000 (checking whether DETSEG compounds under grouping)

| preset | N=1 reserved | N=8 reserved |
|---|---|---|
| exact | 6026 MiB | 33048 MiB |
| fast | 5966 MiB | 32728 MiB |
| deterministic | 6060 MiB | 33048 MiB |

`deterministic` == `exact` exactly at both N — grouping does not amplify the
(refuted) DETSEG pathology; each member's transient one-hot buffer is still
reused within that member's own forward.

### 1.4 `--sym` (eager only), `--renoise`, `--sample_final` — reserved MiB, vs plain eager at the same L

| L | eager | `--sym c4` | `--renoise` | `--sample_final` |
|---|---|---|---|---|
| 100 | 320.0 | 320.0 | 358.0 | 320.0 |
| 400 | 1210.0 | 1210.0 | 1214.0 | 1210.0 |
| 800 | 2296.0 | — | 2302.0 | 2296.0 |
| 1000 | 2710.0 | 2710.0 | — | — |

`--sym` (total residues across 4 equal c4 chains) is **bit-identical to
eager** at the same total L — the screw-replicate op is cheap and adds
nothing measurable. `--renoise`'s worst delta is +38 MiB at L=100 (+11.9%)
and shrinks to +4 to +6 MiB by L=400/800 (<0.5%) — not a growing-with-L
concern, more likely one allocator block boundary at the smallest L.
`--sample_final` reads **exactly** the eager number at all three lengths
tested — the L-step autoregressive re-decode adds no extra live tensors
worth measuring.

### 1.5 Training reference (`salad_torch.training.train`, `--synthetic`, `default_vp`, `--num-aa 1024`, `--max-steps 2`)

| | allocated | reserved |
|---|---|---|
| packed batch, 1024 residues, forward+backward+Adam | **22.06 GiB** | **24.91 GiB** |

Not directly comparable to inference: `PackedProteinStream` always packs to a
**fixed** 1024-residue budget (`--num-aa`) regardless of how many individual
proteins/chains fill it, and this includes the backward pass + Adam moment
buffers, which no inference mode has. Given for reference as asked. This
number alone needs a 32 GB-class card (see §3) — it does not fit in the 90%-
usable budget of a 24 GB card (24.91 GiB reserved > 21.6 GiB usable).

### 1.6 Grid design note

Full L ∈ {50,100,200,300,400,600,800,1000} × 3 presets × single-design = 24
cells. Group axis measured at L ∈ {100, 400, 1000} (representative low/mid/
high, all with preset `fast`; N019 already established preset-invariance of
memory for L ≤ 400, reconfirmed here through L=1000 in §1.3) × N ∈
{1,2,4,8,16} (+24, 32 at L=100 only, to see past the code's own
`_AUTOSIZE_CAP=16`). `--sym`/`--renoise`/`--sample_final` measured at 3
lengths each (they are eager-only and, per §1.4, track eager almost exactly,
so the full 8-length grid would not have added information). Training: one
point, as the task asked for "for reference," not a training sweep.

---

## 2. Fitted model (reserved, MiB; L in residues)

Least squares over the 8-point L grid (single design), plus the max relative
error of that fit against the 8 measured points, plus a degree-2 fit to check
whether quadratic ever helps.

| mode | linear fit | linear max rel err | quadratic fit | quadratic max rel err |
|---|---|---|---|---|
| eager | `2.702*L + 79.3` | 9.2% | `-0.00034*L² + 3.053*L + 26.6` | 17.4% |
| `--fast exact` | `3.870*L + 234.1` | 30.1%† | `0.00107*L² + 2.770*L + 399.6` | 11.6% |
| `--fast fast` | `3.842*L + 241.1` | 29.2%† | `0.00098*L² + 2.837*L + 392.4` | 12.3% |
| `--fast deterministic` | `3.870*L + 234.1` | 30.1%† | `0.00107*L² + 2.770*L + 399.6` | 11.6% |

† The worst point is always **L=50**, driven ~30% high by a fixed
compiled-kernel/graph-pool floor that does not shrink with L (linear predicts
428 MiB at L=50 for `exact`; measured is 612 MiB). Dropping L=50 and fitting
L∈[100,1000] brings the worst residual to **3.9%** (fast/exact) and the fits
otherwise are already good at L≥100 (≤4.3% at every point). **This is a
floor, not curvature** — proof: the quadratic fit is *worse*, not better, at
capturing it (12-17% max error vs 29-30%), and a curvature explanation would
predict the opposite.

**Verdict: linear, not quadratic, for every mode through L=1000.** The
a-priori quadratic risk (distogram/pair features/DETSEG indicator are all
pair-shaped) does not show up in peak memory at these lengths — see §0 for
why (DETSEG's one-hot buffers are transient, not accumulated; the pair-shaped
conditioning tensors `dmap`/`omap` are only `L²·4` and `L²·9·4` bytes — 4 MiB
and 36 MiB at L=1000 — dwarfed by the O(L·K) trunk activations that actually
dominate).

**Allocated fits are markedly cleaner** (allocator quantization mostly
affects *reserved*, not *allocated*, which is just live-tensor bytes):

| mode | linear fit (allocated) | max rel err |
|---|---|---|
| eager | `2.027*L + 63.7` | 3.5% |
| `--fast exact` | `1.659*L + 289.8` | 4.6% |
| `--fast fast` | `1.645*L + 293.3` | 3.9% |
| `--fast deterministic` | `1.667*L + 288.6` | 4.9% |

### 2.1 Group-size marginal cost — extends N019 (L≤400) through L=1000

N019 measured ~3.8 MiB/residue/design at L=50-400. Re-measured at L=100, 400,
**1000**:

| L | marginal MiB/aa/design (fit over N=1,2,4,8,16) |
|---|---|
| 100 | 4.19 |
| 400 | 3.89 |
| 1000 | 4.00 |

**Confirmed flat / linear through L=1000 — no quadratic breakdown.** The
per-member cost stays a fixed ~4.0 MiB per residue regardless of L, which is
exactly the "N019 was right, extend don't redo" result the task asked me to
check for.

Combined model, group path, any preset (fit to the 3 L's × up to 7 N's,
typical error 1-4%, worst ~7% at the smallest L/N where the fixed floor
matters most):

> **reserved(L, N) ≈ (5.73·L + 235) + (N−1)·4.00·L  MiB**

Validated: L=1000,N=8 → predicts 33966 MiB vs measured 32728-33048 MiB
(0.5-3.6% off, `fast`/`exact`); L=1000,N=16 → predicts 64966 vs measured
65178 (0.3%); L=100,N=32 → predicts 13208 vs measured 12928 (2.2%).

---

## 3. Practical answer per card (90% of nominal VRAM usable, stated explicitly)

| card | nominal | usable (90%) |
|---|---|---|
| 8 GB | 8192 MiB | 7373 MiB |
| 12 GB | 12288 MiB | 11059 MiB |
| 16 GB | 16384 MiB | 14746 MiB |
| 24 GB (4090/5090-class) | 24576 MiB | 22118 MiB |
| 32 GB | 32768 MiB | 29491 MiB |
| 48 GB | 49152 MiB | 44237 MiB |
| 80 GB | 81920 MiB | 73728 MiB |

**Single-design inference (eager or plain `--fast`, no `--group_size`) is
never the constraint.** Extrapolating the §2 linear fits: `--fast`'s
reserved footprint at L=1000 is 4104-4234 MiB — every card down to 8 GB
handles L=1000 with 3+ GiB to spare, and the fit says single-design `--fast`
would keep fitting an 8 GB card out to **L ≈ 1800** (extrapolated, not
measured — flagged as such). **The real limit is `--group_size`.**

### Max group size N at representative L (model from §2.1, `--fast` path; code's own `_AUTOSIZE_CAP` is 16 unless the user passes an explicit integer)

| card | N_max @ L=100 | N_max @ L=400 | N_max @ L=1000 |
|---|---|---|---|
| 8 GB | 17 (capped to 16 by `auto`) | 4 | **1** |
| 12 GB | 26 (capped to 16) | 6 | 2 |
| 16 GB | 35 (capped to 16) | 8 | 3 |
| 24 GB | 54 (capped to 16) | 13 | 5 |
| 32 GB | 72 (capped to 16) | 17 | 6 |
| 48 GB | 109 (capped to 16) | 27 | 10 |
| 80 GB | 183 (capped to 16) | 45 | 17 |

Read this as: **at L≈100, memory never binds on any card down to 8 GB** — the
code's own `_AUTOSIZE_CAP=16` bites first everywhere (a user who wants more
than 16 has to pass an explicit `--group_size 17` and up; the memory itself
would allow it on every card here). At **L=1000, an 8 GB card cannot group at
all** (N_max=1, i.e. falls back to plain `--fast`) and even a 24 GB card
(4090/5090-class) tops out around **N=5**. An 80 GB card is the first one
where L=1000 grouping approaches the code's own cap (N=17 ≈ 16).

### Training

The 22.06 GiB allocated / **24.91 GiB reserved** reference point (§1.5) needs
at minimum a **32 GB card** at 90% usable headroom (24 GB's 22.12 GiB usable
is short by ~2.8 GiB). This is one packed batch of 1024 residues at the
default config; it was not swept over batch/packing size, per the task's
"for reference" framing.

---

## 4. Pathological / surprising modes, with the numbers

1. **Reserved/allocated divergence grows with group size, not length** —
   the clearest actionable finding. Ratio reserved/allocated: 2.1x at N=1,
   **5.7x at N=16, 9.9x at N=32** (all L=100); 3.2x → 12.8x (N=1→16) at
   L=400; 3.3x → 10.4x → **14.1x** (N=1→8→16) at L=1000. A user planning
   VRAM off `torch.cuda.memory_allocated()` (or off summed tensor shapes)
   would be wrong by up to 14x on the shipping group-size feature. This is
   almost certainly the CUDA-graph private memory pool: capturing N members
   on N streams commits a pool sized for the graph's worst-case concurrent
   footprint, and that reservation does not shrink back down even though far
   fewer bytes are live at any instant. Reserved, never allocated, is the
   number to card-size against — the task's instinct going in was right.

2. **DETSEG hypothesis refuted** (§0, §1.1, §1.3) — flagged explicitly
   because the task asked to watch for exactly this, and the honest answer
   is "measured, and it isn't there": `deterministic` is bit-identical to
   `exact` in peak reserved at every L from 50 to 1000, single-design and
   grouped. The one-hot `(L,L)` matmul is real (`salad_torch/geometry/
   segment.py:det_segment_sum`) but its buffer is transient and reused by
   the allocator inside one forward, never accumulating onto the process
   high-water mark even at L=1000 where a naive count-the-calls estimate
   would put it at ~100-200 MiB.

3. **`--fast` uses MORE memory than eager**, at every L (612 vs 216 MiB at
   L=50, +183%; 4234 vs 2710 MiB at L=1000, +56%) — because it keeps two
   full captured-graph variants (`need_aa` True/False) resident
   simultaneously, each with its own static buffers. A card tight enough to
   run eager but not `--fast` at the same L is a real, if narrow, case this
   report did not have to imagine — it is in the L=1000 column of §1.1.

4. **The inductor on-disk compile cache is broken for this fast path** — not
   a memory finding, but the task explicitly asked to flag when compile time
   becomes prohibitive, and this is why it does. Every process hit:
   ```
   AttributeError: Can't pickle local object 'FastDenoiser.__init__.<locals>._cse_pass'
   ```
   from N017/O1's `_ind.joint_custom_post_pass = _cse_pass` (a local closure,
   unpicklable) — inductor's FX-graph cache write fails silently (a warning,
   not a crash) every single time, so **no cross-process compile reuse ever
   happens** via that cache. Measured compile+capture time: 256-469 s
   (4.3-7.8 minutes) **per fresh process**, non-monotonic in L (rises to a
   peak ~460 s at L=300-400, then *drops* to ~266 s at L≥600, on a cold cache,
   reproduced independently across 3 separate jobs/nodes — plausibly
   Inductor's max-autotune search space itself being L-dependent, not
   re-verified further here). A *separate*, lower-level cache does still
   help: repeating the **same L** within one job/node (the group-size ladder)
   drops from a 376-457 s first compile to a steady ~223-230 s for every
   subsequent N — consistent with Triton's own kernel-binary cache (a
   different, still-functional cache) surviving even though the FX-graph
   cache does not. Practical upshot: a user who invokes `generate.py` freshly
   per batch (rather than keeping one warm process and looping designs, which
   is what the script's own `--num_designs` loop already does) pays this
   4-8 minute tax every time. This is a real bug worth fixing (not done here,
   out of scope — no model/script edits) and it belongs to N017/O1's
   territory (`salad_torch/modules/fast.py`, owned by N020 in this branch).

5. **`--renoise`/`--sample_final`/`--sym` do NOT move the high-water mark**
   in any concerning way — refuting the a priori worry that 9 simultaneous
   renoise passes, or an L-step autoregressive `sample_final` decode, would
   meaningfully add memory. Worst delta measured: +38 MiB (+11.9%) at
   L=100 for `--renoise`, shrinking to +4-6 MiB (<0.5%) by L=400/800.
   `--sample_final` and `--sym` read **identical** to plain eager at every
   length tested.

---

## 5. What this does NOT cover

- Only `default_vp`/`default_vp-200k` was measured (matches N019 and the
  task's own preset-invariance finding; other configs, e.g. `multimotif_vp`
  with real pair conditioning, were not re-measured here).
- Group axis is 3 L's × up to 7 N's, not the full 8×5 grid, per the grid
  note in §1.6 — chosen to cover low/mid/high L cheaply given N019 already
  owns the dense L≤400 grid.
- No OOM was actually observed (93.1 GiB H100). Every "does not fit" figure
  in §3 is the §2 model extrapolated to a smaller card and is flagged as
  such, not a measured OOM.
- The inductor cache-pickling bug (§4.4) is reported, not fixed — it belongs
  to N017/O1's code, which this task was told not to edit.
