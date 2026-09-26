# N019 — N independent designs in one CUDA graph

**The headline, honestly: N016's claim reproduces on the production path, at
1.81x rather than 2.26x** (`fast`, L=100, `scripts/generate.py`, end to end:
**0.603 -> 0.334 s per design**). Every design in a group of 8 is
**bit-identical** to the same design generated alone. The win is an
H100-at-L=100 number and shrinks to 1.16x at L=400 and, by extrapolation,
towards 1.0x on a small consumer card - which is why `--group_size auto`
measures the group size on whatever card it is running on instead of
evaluating a formula.

Jobs: 904093/904175 (parity), 904098 (smoke), 904122-904131 (N x L curves),
904132/904133/904175 (identity gate), 904150-904153 (end to end).

All numbers: H100 (132 SM, sm_90, 93.1 GiB, driver 610.57.04), torch
2.5.1+cu121, `default_vp-200k`, one length **and one group size** per process,
detached snapshot at `/hnvme/workspace/v181bb10-salad/n019tree`.

---

## 1. What was built

`--group_size N` on `scripts/generate.py`. N independent designs are advanced
through **one** captured CUDA graph, each on its own stream.

This is **not** a batch axis. salad packs a batch into one flat residue axis
with `batch_index`/`chain_index` companions; doing that here would change
every design's arithmetic (neighbour sets, segment reductions) and is a
different feature. What happens instead:

* each member keeps its **own complete set of static input buffers** and its
  own copy of `Diffusion.forward_core`;
* the members are recorded into one `torch.cuda.CUDAGraph` on N streams, so
  one `replay()` submits all of them and the driver overlaps whatever it can;
* **no tensor, no reduction and no RNG draw is shared between members**, so
  each design computes exactly what it would have computed alone.

Three pieces:

| file | what |
|---|---|
| `salad_torch/modules/fast.py` | `GroupFastDenoiser`, `memory_bound_group`; `FastDenoiser.forward` split into `_prepare_inputs` / `_run` / `_postprocess` so the group can interleave N members' host work around one replay; `pin_condition` |
| `salad_torch/diffusion/sampling.py` | `sample_group` — `sample`'s loop with every per-design statement moved inside a `for i in range(n)` |
| `scripts/generate.py` | `--group_size`, `run_grouped`, `make_design_data`, `design_generator` |

### 1.1 Why the RNG had to change, and what that costs

The ordinary loop runs every design off **one** generator in sequence, so
design 1's noise depends on how many draws design 0 made. That ordering cannot
survive lockstep advancement. The grouped path therefore derives an
independent stream per design (`design_generator`: `seed * 1000003 + i`).

This is what makes the gate checkable at all: design *i* of a
`--group_size 8` run and design *i* of a `--group_size 1` run are fed the same
stream, so they must come out the same. It also means a grouped run does not
reproduce a *legacy* run's designs — same distribution, different draws. The
legacy path is untouched and still the default.

### 1.2 Host-side work, and the sync that had to go

`FastDenoiser._prepare_inputs` content-hashes the conditioning tensors on
**every step** (`.cpu()` per tensor — a device→host sync per step) to key the
`prepare_condition` cache. Affordable against one 3.5 ms forward; not
affordable N times inside a grouped step, where the host has to stay ahead of
a single replay. `pin_condition` computes the conditioning once per design and
drops the per-step hash. That is exactly equivalent, not an approximation:
`sample()` writes only `pos`/`seq`/`atom_pos`/`aatype`/`t_pos`/`t_seq`/
`latent` into `data` and the hash keys on none of them, so from step 1 on the
cache always hit anyway and the hash was pure cost.

---

## 2. Portability — this must not be an H100 optimisation

The mechanism is **idle SMs**. A card with fewer SMs, lower clocks or less
bandwidth has less idle headroom to reclaim, so the win necessarily shrinks.
Three things follow, and all three are in the code rather than in this
document:

**(a) N is never a constant.** `--group_size` is a runtime parameter.
`--group_size auto` does not evaluate a formula fitted to this node — it
**measures** on whatever device it is running on: it captures the graph at
1, 2, 3, 4, 6, 8, … (`_AUTOSIZE_LADDER`), times the replay the sampler spends
~99 % of its steps in, and stops when a rung buys less than 3 % per design or
when free VRAM says stop. A rung costs a *capture*, not a recompile — the
compiled forward is shared across members — so the whole search is seconds
against a one-time compile of a minute or more. An analytic rule calibrated on
a 132-SM H100 is exactly the specialisation this is meant to avoid, and it
would be wrong in an unknown direction on a 4090, a 5090, or a card that does
not exist yet.

**(b) The memory bound is measured, not estimated.**
`GroupFastDenoiser._probe_member_cost` runs one compiled forward on the real
model and takes the allocation high-water; `memory_bound_group` turns that
plus `torch.cuda.mem_get_info()` into a member ceiling
(`1 + free * 0.6 / per_member`). On a 24 GB card this, not occupancy, is
normally what decides N.

**(c) Degrading to 1 is a normal outcome, not a failure.** `group_size=1`
captures a **single-stream body byte-for-byte identical to `FastDenoiser`'s**,
so "no grouping" is the same code path rather than a fallback branch. A
capture that runs out of memory halves the group and retries, down to 1.

**Nothing here is architecture-gated.** No TMA, no wgmma, no cluster launch,
no inline PTX, no hand-written CUDA. The only primitives are
`torch.cuda.Stream`, `torch.cuda.CUDAGraph` and `torch.compile`, all generic,
and the member forward is whatever Inductor autotunes for the card it lands
on.

**Flagged as untestable here:** this environment is torch 2.5.1+cu121, which
cannot target sm_120 at all (Blackwell needs CUDA 12.8+ / torch ≥ 2.7). The
Blackwell claim is therefore an argument from portability of the primitives,
not a measurement. Two places where a newer torch/Triton could behave
differently and should be re-measured rather than assumed:
1. multi-stream capture inside `torch.cuda.graph` — legal and stable in 2.5,
   but the allocator's graph-pool bookkeeping has changed across releases;
2. Inductor's kernel choices for the member forward, which is the point of
   leaving them to Inductor, but which means the *shape* of the N curve is a
   per-machine fact and is why `auto` measures.

---

## 3. The gate: is design *i* of a group of N the same design?

**PASS, in its strongest form: bit-identical, 8/8 designs, both presets, over
the full production schedule, through `scripts/generate.py` itself.**

Jobs 904132 / 904133, plus 904175 on the commit candidate. `x_identity.py`
runs the real `generate.main()` and side-saves each written design's
full-precision `pos`, so the comparison is not truncated to the PDB's three
decimals.

### 3.1 `SALAD_OPT_DETSEG=1` — the arm that decides it

| | `exact` | `fast` |
|---|---|---|
| designs compared (`--group_size 1` vs `--group_size 8`) | 8 | 8 |
| **bit-identical** | **8/8** | **8/8** |
| max abs coordinate difference | **0.000e+00 A** | **0.000e+00 A** |
| `aatype` differences | **0/800** | **0/800** |

Under the deterministic segment reduction the forward is reproducible by
construction (N013), and under it a grouped design is **bit-for-bit** the
design it would have been alone. Not "within a floor" — identical.

This arm is what carries the gate, and the argument is tighter than the
number alone. `SALAD_OPT_DETSEG` changes exactly one thing: `index_sum`'s
`scatter_add` atomics become a deterministic one-hot matmul. If grouping
perturbed *anything* other than the order in which those atomics land —
a different kernel, a different neighbour set, a shared buffer, a leaked
reduction, a stream-dependent cuBLAS algorithm — DETSEG would not restore bit
identity, because DETSEG does not touch any of those. It restores it
completely. **So grouping's only effect on the arithmetic is the atomic
ordering**, which is the same and only source of the single-design path's own
run-to-run noise.

### 3.2 The shipping configuration (atomics) — measured, and it cannot discriminate

The same job measures the self-divergence floor the honest way: the identical
`--group_size 1` command, identical seed, run twice, so the only free variable
left is `scatter_add` ordering.

| `exact`, 8 designs, L=100, 401 steps | max abs Δpos | `aatype` differing |
|---|---|---|
| **floor** (g1 vs g1, identical command) | 989 / 1316 / 1248 A worst three | **381/800 (48 %)** |
| grouped (g1 vs g8) | 978 / 1427 / 1294 A worst three | 441/800 (55 %) |

**The floor is already at the "completely different design" level**, so this
arm has no discriminating power and I am not claiming it as evidence either
way. That is not a property of grouping; it is the documented behaviour of
this port's CUDA sampler (N009: "the CUDA denoiser is NOT reproducible,
`index_mean` `scatter_add`, 1.755e-02 self-noise"), amplified by 401 steps of
a chaotic rollout. A rerun of the *same command* already changes 48 % of the
residues.

What can honestly be said: the grouped difference is **the same size as the
floor** (coordinate ratio 978/989 ≈ 0.99, 1294/1248 ≈ 1.04 on the designs
that diverge; the `aatype` counts, 59–75 grouped vs 58–75 floor for `exact`,
overlap design for design). Reporting the ratio to an identical-input rerun
rather than an absolute deviation is the lesson this project already paid for
once.

`fast` behaves the same way (floor 21–74 residues, grouped 2–90).

### 3.3 The standard suite

Job 904093, re-run as job 904175 on the commit candidate:

| test | result |
|---|---|
| `test_model_parity` `default_vp` | max 0.00002 A |
| `test_model_parity` `default_vp_scaled` | max 0.00002 A |
| `test_model_parity` `default_ve_scaled` | max 0.00002 A |
| `test_model_parity` `multimotif_vp` | max 0.00020 A |
| `test_geometry_parity` | exit 0 |
| `test_stack_parity` | exit 0 |
| `test_encoder_parity` | exit 0 |
| `test_sampling_parity` | exit 0 |
| | **`SUITE_EXIT=0`** |

Every floor is exactly the documented one, so splitting `FastDenoiser.forward`
into `_prepare_inputs` / `_run` / `_postprocess` and adding `pin_condition`
moved nothing on the single-design path.

---

## 4. The N and L curves

Jobs 904122–904131. One `(preset, L, N)` per **process**; the inductor cache
is shared within a job so only the first process compiles. Every time is
measured inside a captured replay or around a real `sample_group` loop —
never around an eager op.

`replay` is the captured graph alone; `host step` is the full production step
with `apply_diffusion`, the buffer copies, the output clones and
`_postprocess` included — the number that actually sets seconds per design.
Both are **per design** (total / N).

### 4.1 `exact`, host-inclusive step, speedup vs N=1

| N | L=50 | L=100 | L=150 | L=200 | L=300 | L=400 |
|---|---|---|---|---|---|---|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.49 | 1.40 | 1.34 | 1.27 | 1.21 | 1.13 |
| 3 | 1.76 | 1.58 | 1.47 | 1.37 | **1.27** | **1.17** |
| 4 | 1.98 | 1.70 | 1.56 | 1.43 | 1.30 | **1.17** |
| 6 | 2.22 | 1.82 | 1.63 | **1.47** | **1.30** | 1.17 |
| 8 | 2.37 | 1.89 | **1.65** | 1.48 | 1.29 | 1.16 |
| 12 | 2.49 | 1.92 | 1.64 | 1.46 | 1.28 | 1.15 |
| 16 | **2.61** | **1.94** | 1.64 | 1.46 | 1.28 | 1.15 |

### 4.2 `fast`, host-inclusive step, speedup vs N=1

| N | L=50 | L=100 | L=200 | L=400 |
|---|---|---|---|---|
| 2 | 1.46 | 1.37 | 1.23 | 1.14 |
| 3 | 1.70 | 1.53 | 1.32 | **1.17** |
| 4 | 1.90 | 1.65 | 1.38 | 1.17 |
| 6 | 2.13 | 1.75 | **1.40** | 1.16 |
| 8 | 2.26 | 1.81 | 1.40 | 1.14 |
| 12 | 2.36 | 1.81 | 1.38 | 1.14 |
| 16 | **2.42** | **1.83** | 1.38 | 1.13 |

### 4.3 Where the optimum sits, and what `auto` chose

| preset | L | best N (measured) | speedup there | `auto` picked | `auto`'s speedup |
|---|---|---|---|---|---|
| exact | 50 | 16 (still rising) | 2.61 | 12 | 2.50 |
| exact | 100 | 12–16 | 1.94 | **8** | 1.89 |
| exact | 150 | 8 | 1.65 | **6** | 1.63 |
| exact | 200 | 8 | 1.48 | **6** | 1.47 |
| exact | 300 | 4–6 | 1.30 | **3** | 1.27 |
| exact | 400 | 4 | 1.17 | **3** | 1.16 |
| fast | 50 | 16 (still rising) | 2.42 | 12 | 2.37 |
| fast | 100 | 16 | 1.83 | **8** | 1.81 |
| fast | 200 | 6 | 1.40 | **4** | 1.38 |
| fast | 400 | 3 | 1.17 | **3** | 1.17 |

**`auto` lands on the knee at every one of the ten points**, and always within
5 % of the best pinned N — while choosing a *smaller* group, i.e. less memory,
than the flat tail. Its 3 % stopping rule is doing exactly its job: at L=400
it stops at 3 where 4 is 0.5 % better, which is noise.

The curve **saturates and then very slightly regresses** at every L ≥ 150 —
past the knee the extra members add capture memory and host work without
adding overlap. There is no runaway; picking N too large costs a few percent,
not a cliff.

### 4.4 The other half of the curve: peak VRAM

Peak **reserved** bytes, whole process (so it includes the one-time compile
and, for `auto`, the ladder search — which is why `auto` reads slightly above
a pinned run at the same N).

| L | base (N=1) | marginal per extra design | N=4 | N=8 | N=16 |
|---|---|---|---|---|---|
| 50 | 878 MiB | 213 MiB | 1.08 GiB | 2.07 GiB | 3.98 GiB |
| 100 | 1046 MiB | 386 MiB | 1.75 GiB | 3.41 GiB | 6.67 GiB |
| 150 | 1246 MiB | 586 MiB | 2.53 GiB | 4.97 GiB | 9.80 GiB |
| 200 | 1586 MiB | 778 MiB | 3.32 GiB | 6.54 GiB | 12.94 GiB |
| 300 | 2082 MiB | 1150 MiB | 4.81 GiB | 9.52 GiB | 18.88 GiB |
| 400 | 2664 MiB | 1522 MiB | 6.31 GiB | 12.53 GiB | 24.89 GiB |

Identical for both presets — same shapes. The marginal cost is linear in L at
**~3.8 MiB per residue per design**, which is what makes
`memory_bound_group`'s measured `per_member_bytes` a sound basis.

**On a 24 GB card**, with ~22 GiB usable: at L=100 even N=16 (6.7 GiB) fits
with room to spare, so *speed*, not memory, picks N. At L=400 the useful N is
3–4 (6.3 GiB) — also comfortable; memory only binds at N ≳ 10, well past the
point where the curve has gone flat. **So on a 4090-class card this feature
should never be memory-limited at a group size worth using**, and
`_AUTOSIZE_MEM_FRACTION = 0.6` plus the OOM-halving retry cover the rest.

---

## 5. Seconds per design — before and after

Jobs 904150–904153, the real `scripts/generate.py`, 8 designs, wall clock,
compile/capture excluded (it is one-time).

| preset | L | **today** (`--fast`) | `--group_size 1` | `--group_size N` | **speedup** |
|---|---|---|---|---|---|
| `fast` | 100 | **0.603 s** | 0.603 s | **0.334 s** (N=8) | **1.81x** |
| `exact` | 100 | **1.34 s** | 1.385 s | **0.740 s** (N=8) | **1.87x** |
| `fast` | 400 | **1.36 s** | 1.371 s | **1.182 s** (N=4) | **1.16x** |
| `exact` | 400 | **2.95 s** | 2.974 s | **2.453 s** (N=4) | **1.21x** |

`--group_size 1` reproduces the legacy path's time to within 3 %, which is the
check that the grouped machinery costs nothing when it is not grouping.

### 5.1 The honest accounting against N016's 2.26x

N016 (R4) projected `fast` at L=100 going from 0.58 to **0.26 s/design**
(2.26x). The production path delivers **0.334 s/design, 1.81x**. The claim
reproduces in kind and in mechanism, but it is **~20 % smaller**, and it is
worth saying exactly where the difference is:

| | `fast` L=100 | ratio |
|---|---|---|
| N016, compiled core, k=8 vs k=1, its own harness | 1.3895 vs 3.007 ms | **2.26x** |
| this work, production capture, replay only | 1.3961 vs 2.8041 ms | **2.01x** |
| this work, production step, host work included | 1.6566 vs 3.0017 ms | **1.81x** |
| this work, end to end, `generate.py`, s/design | 0.334 vs 0.603 s | **1.81x** |

Two losses, both real and both explained:

1. **2.26x → 2.01x is the baseline, not the grouping.** N016 flagged this
   itself: its k=1 was 3.696 ms against the production replay's 3.395 ms
   (fresh argument sets, graph-pool allocation), so its ratio was taken
   against an inflated denominator. Measured against the *production* N=1
   the same 8-way capture gives 2.01x. The N=8 numbers themselves agree to
   0.5 % (1.3961 vs 1.3895 ms), so nothing was lost in the port — the
   reference moved.
2. **2.01x → 1.81x is per-design host work that does not overlap.** The step
   costs 0.198 ms/design of host work at N=1 and 0.26 ms/design at N=8:
   `apply_diffusion` eager, ~19 `copy_` into the static buffers, the output
   clones, `_postprocess`. Eight designs' worth (2.1 ms) sits against a
   13.3 ms replay, so the CPU stays ahead overall, but it is not free. This
   is the term `pin_condition` already removed the worst of (a device→host
   sync per design per step); what remains is genuinely per-design work.

**So: the finding is verified on the production path and it is real. The
number is 1.81x / 0.334 s per design for `fast` at L=100, not 2.26x /
0.26 s.** A user asking for 100 designs of 100 aa goes from 60 s to 33 s.

---

## 6. How this scales DOWN on a smaller GPU

The mechanism is idle SMs, so the win must shrink on a card with less spare
capacity. The L sweep is the best empirical handle I have on that, because
**increasing L and shrinking the card push the same ratio**: both raise the
fraction of the device one forward occupies. Measured, `exact`, best N:

| relative work per forward (L/100) | 0.5 | 1 | 1.5 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| measured speedup | 2.61 | 1.94 | 1.65 | 1.48 | 1.30 | 1.17 |

which is fitted to within 0.14 across the whole range by

> **S ≈ 1 + 0.88 / w**,  w = (work per forward) / (work at L=100 on a 132-SM H100)

Reading a smaller card as a larger `w` (capacity proxy: SM count relative to
the H100's 132) gives, **as an extrapolation with no measurement behind it**:

| card | SMs | est. at L=100 | est. at L=400 |
|---|---|---|---|
| H100 (measured) | 132 | **1.94x** | **1.17x** |
| RTX 5090 | 170 | ~2.1x | ~1.3x |
| RTX 4090 | 128 | ~1.85x | ~1.2x |
| RTX 5080 | 84 | ~1.6x | ~1.14x |
| RTX 4070 | 46 | ~1.3x | ~1.08x |
| RTX 4060 | 24 | ~1.16x | ~1.04x |

**Direction, stated plainly: the win falls monotonically with SM count and
with protein length, and approaches 1.0x — i.e. the existing single-design
path — on a small card or a long protein.** A 4090 has nearly the H100's SM
count, so at L=100 it should see most of the win; a 46-SM laptop-class part
should expect ~1.3x and should mostly be running N=2–3.

**Do not treat 2.22x, or even 1.94x, as a portable number. It is an H100,
L=100 number.** Three reasons not to lean on the table above either: SM count
ignores clocks, memory system and driver; the binding term here is launch
latency rather than SM occupancy, and those need not scale together; and this
node cannot run Blackwell at all. That is precisely why `--group_size auto`
**measures** instead of evaluating a formula — the user never has to trust
this extrapolation, and on a card where the answer is 1, `auto` returns 1 and
they get today's behaviour.

---

## 7. Caveats

**C1 — sm_120 is argued, not measured.** torch 2.5.1+cu121 cannot target
Blackwell (needs CUDA 12.8+ / torch ≥ 2.7). Everything here is generic
`torch.cuda.Stream` + `torch.cuda.CUDAGraph` + `torch.compile`, with no
arch-gated code, so the *mechanism* ports; the *number* must be re-measured,
and `auto` does that on first run. Two places a newer torch/Triton could
differ: the allocator's graph-pool bookkeeping around multi-stream capture,
and Inductor's kernel choices for the member forward (which is the point of
leaving them to Inductor, and also why the shape of the N curve is a
per-machine fact).

**C2 — the atomics identity arm is uninformative, by construction.** Its
floor is already at 48 % of residues on a rerun of the same command. The gate
rests on the DETSEG arm (§3.1) and the argument in §3.2 that DETSEG isolates
exactly the one thing grouping can touch. A reader who wants a
floor-normalised number on the shipping path cannot have one from this
model — that is a property of the CUDA sampler (N009), not of N019.

**C3 — the RNG stream changed for grouped runs.** `design_generator` gives
design *i* a seed-derived stream instead of a position in one shared
sequence, so a grouped run does not reproduce a *legacy* run's designs (same
distribution, different draws). This is unavoidable in lockstep and is what
makes the gate checkable. The legacy path is untouched and still the default.

**C4 — no design-quality arm.** The gate is bit-identity, which is stronger
than any structural test, so `--group_size` cannot change the design
distribution by construction. No ProteinMPNN/folding check was run and none is
needed for this change.

**C5 — `peak_reserved` is a process high-water**, so it includes inductor's
compile scratch and, for `auto`, the ladder search. The *marginal* figures in
§4.4 (the N=1 → N=16 slope) are the ones to size a card with.

**C6 — one node.** All timings are h11-17/h11-24/h13-10/h13-11, one H100
each, one length and one group size per process. No cross-node replication of
the speed numbers; the gate was reproduced on two different jobs and two
presets.

**C7 — `--group_size` is refused with `--sym`, `--renoise`, `--sequence_only`
and `--sample_final`**, each of which inserts per-design host work between the
noising and the denoising that one capture cannot see. Those flags get the
ordinary path, which is unchanged.
