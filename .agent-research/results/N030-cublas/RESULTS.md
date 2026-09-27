# N030 - the cuBLAS side: can inductor's matmul templates be made bit-exact?

Tree: `n006-manual-cudagraph` @ 98bc61b, detached worktree
`/hnvme/workspace/v181bb10-salad/n030`. (The brief called this item N027;
N027/N028/N029 were taken by the two concurrent sibling agents before this
item was created.)

Baseline, L=100, `default_vp`, preset `exact`, lean step
(`need_aa=False`, `pair_cond=False`), H100, job 907130 arm `off`:

**lean 3.3705 ms**, 782 kernels = 392 Triton (1059.8 us) + **352 cuBLAS
(2138.7 us)** + 36 memcpy + 2 other, GPU busy 3244.2 us.

cuBLAS is 65.9 % of GPU-busy time at **6.08 us per launch**.

## 0. What the 352 cuBLAS launches actually are (job 907130, arm `off`)

Per graph replay, by kernel NAME - this is the first time the launches have
been broken down by what cuBLAS actually ran, rather than counted:

| kernel | launches | us | us/launch |
|---|---|---|---|
| `sm80_xmma_gemm_f32f32..._nn_..._tilesize32x32x8` | 120 | 829.2 | 6.91 |
| `..._tt_..._tilesize64x64x8` | 18 | 268.8 | 14.93 |
| `..._nn_..._tilesize64x64x8` | 24 | 268.2 | 11.18 |
| `..._tt_..._tilesize32x32x8` | 24 | 169.1 | 7.05 |
| `cutlass_80_simt_sgemm_64x64_8x5_nn_align1` | 30 | 141.9 | 4.73 |
| **`gemv2N_kernel<...>`** | **24** | **94.6** | **3.94** |
| **`splitKreduce_kernel<32,16,...>`** | **38** | **67.1** | **1.77** |
| `..._tn_..._tilesize32x32x8` | 18 | 61.9 | 3.44 |
| `..._nt_..._tilesize32x32x8` | 12 | 49.5 | 4.13 |
| **`internal::gemvx::kernel<...>`** | **12** | **46.6** | **3.88** |
| `cutlass_80_simt_sgemm_128x32_8x5_nn_align1` | 7 | 40.5 | 5.79 |
| `..._nn_..._tilesize64x32x8` | 6 | 38.7 | 6.45 |
| `..._nn_..._tilesize64x64x8` (2nd instantiation) | 12 | 35.7 | 2.98 |
| `..._nn_..._tilesize32x32x8` (2nd instantiation) | 6 | 16.0 | 2.67 |
| `cutlass_80_simt_sgemm_128x64_8x5_nn_align1` | 1 | 10.9 | 10.9 |

Two answers to "are all 352 launches necessary":

* **38 of them are not GEMMs at all.** `splitKreduce_kernel` is the second
  half of a cuBLAS split-K GEMM - the reduction over the K-partitions. So
  cuBLAS is *itself* spending 38 extra launches (67.1 us) on 38 of these
  problems, and - separately important for parity - those 38 GEMMs are
  ALREADY reassociated relative to a sequential K loop in the model as
  shipped today.
* **36 more are GEMV**, i.e. cuBLAS decided the problem is a matrix-vector
  product and is paying 3.9 us of launch latency each for it.

## 1. The GEMM shape census (static, from inductor's generated module)

All 301 `extern_kernels.bmm` calls in the L=64 CPU output code
(`N020/cpu_output_code_einsum.py.gz`), grouped by (B, M, K, N):

| calls | B | M | K | N | MFLOP/call |
|---|---|---|---|---|---|
| 30 | 1 | 64 | 128 | 512 | 8.4 |
| 17 | 64 | 3 | 3 | 20 | 0.02 |
| 15 | 1 | 64 | 128 | 768 | 12.6 |
| 15 | 1 | 64 | 128 | 576 | 9.4 |
| 15 | 64 | 3 | 3 | 64 | 0.07 |
| 15 | 1 | 64 | 1024 | 128 | 16.8 |
| 10 | 1 | 64 | 512 | 128 | 8.4 |
| 9 | 64 | 3 | 3 | 1280 | 1.5 |
| 9 | 1 | 64 | 60 | 256 | 2.0 |
| 9 | 1 | 64 | 256 | 128 | 4.2 |
| 6 | 1 | 4096 | 400 | 64 | 209.7 |
| ... | | | | | |

**~90 of the ~300 calls are 3x3 frame rotations** (`M=3, K=3`, batch 64-18432,
0.02-1.5 MFLOP each) - the `rigid.py`/`frames.py` algebra N017 attributed
18 % of Triton time to, here paying a ~6 us cuBLAS launch apiece. That is the
single largest identifiable block of pure launch waste on the cuBLAS side and
it motivated arm `decomp32` below.

## 2. Whole-forward speed, L=100, preset `exact`, lean step (job 907130 + 907226)

One arm per python, one length per process. `off` is the shipped default.

| arm | lean ms | vs base | full ms | kernels | triton | templates | cuBLAS | cuBLAS us | GPU busy | compile s |
|---|---|---|---|---|---|---|---|---|---|---|
| `off` (shipped) | 3.3688 | - | 4.1608 | 782 | 392 | 0 | 352 | 2138.7 | 3244.2 | 258 |
| `all` (N020's arm) | 2.8886 | **-14.3 %** | 3.5160 | 648 | 330 | 259 | 57 | 304.4 | 2747.7 | 660 |
| `decomp32` (NEW) | 2.8108 | **-16.6 %** | 3.4779 | 752 | 458 | 0 | 268 | 1445.8 | 2680.3 | 256 |

`all` reproduces N020's -14.0 % exactly, on a tree whose baseline has moved
(P7 disabled: 352 cuBLAS launches now, not 301).

## 3. The idea: verify bit-exactness instead of trusting the timing race

`salad_torch/gemm_exact.py`. It wraps the `autotune_select_algorithm` name
inside `torch._inductor.kernel.{mm,bmm}` and, before inductor picks a winner,
runs every Triton candidate and the ATen (cuBLAS) candidate once on the same
random inputs at the real shape and keeps only the candidates whose output is
**bitwise** equal to cuBLAS's. Inductor then autotunes among the survivors as
usual.

Two fp32 GEMM kernels differ only if they differ in the ORDER (or the fusion)
of their multiply-accumulates, and that is a property of the kernel, fixed at
compile time and independent of the data. One random draw at the real shape
therefore compares those two orders across every element of the output at once
(e.g. 6400x64 = 410k independent length-K dot products). The check is
`torch.equal` on the raw bytes, not a tolerance.

This is the PORTABLE form of the per-shape allowlist the brief asked for: the
verification re-runs at compile time on whatever card is present, so a card
where no Triton candidate matches keeps today's cuBLAS kernels and loses the
speedup rather than the exactness. Nothing in the file names an architecture,
a block size, a warp count or a stage count.

### 3a. What the verification finds (L=100, job 907149, arm `exact`)

**401 GEMM sites, 400 verified** (1 hit the fail-safe and kept ATen),
**270 of them (67 %) have at least one bitwise-exact Triton candidate**, and
**3769 of 5756 Triton candidates (65 %) are bitwise equal to cuBLAS**.

By shape - and this is the part a hand-written allowlist would have got wrong:

| shape (B, M, K) x (B, K, N) | exact / candidates |
|---|---|
| (100, 3, 3) x (100, 3, 20 / 64 / 192 / 960 / 1280 / 1920 / 2560) | 5/5 .. 12/12 |
| (6400, 3, 3) x (6400, 3, 3), (4800, 3, 3) x ... | 5/5 |
| (1, 6400, 9) x (1, 9, 64) | 15/15 |
| (1, 6400, 60) x (1, 60, 64) | 18/18 |
| (1, 6400, 64) x (1, 64, 128) | 19/19 |
| (1, 6400, 64) x (1, 64, 8) | 16/16 |
| (100, 8, 64) x (100, 64, 64) | 12/12 |
| (1, 6400, 128) x (1, 128, 64) | 18/18 |
| (1, 100, 128) x (1, 128, 576 / 768) | 18/19 |
| (1, 6400, 240) x (1, 240, 64) | 17/18 |
| **(1, 100, 128) x (1, 128, 128)** | **0/19** |
| **(800, 1, 32/48/64) x (800, ..., 24/32/48/64)** | **0/8 .. 0/10** |
| (1, 6400, 400) x (1, 400, 64) | 0/18 |
| (1, 100, 512) x (1, 512, 128) | 0/19 |
| (1, 100, 1024) x (1, 1024, 128) | 0/19 |
| (1, 100, 1725) x (1, 1725, 512) | 0/19 |

The K <= 240 rule that N016's E4 suggested is broadly right, and **it has two
exceptions that a static allowlist would have shipped as bugs**:

* every **M=1** shape is 0/N - cuBLAS runs a GEMV kernel there (the 36 GEMV
  launches in section 0), whose reduction order no matmul template reproduces;
* **(1, 100, 128) x (1, 128, 128)** is 0/19 while the same K=128 with N=576 or
  N=768 is 18/19 - cuBLAS switches to a split-K kernel at that N (the 38
  `splitKreduce` launches in section 0), and split-K is a reassociation.

Both are found by measurement and neither is predictable from K.

## 4. Is the per-shape verified-exact filter bit-exact END TO END? NO.

The theory (section 3's docstring in `gemm_exact.py`) is that verifying every
survivor bitwise-equal to cuBLAS at compile time should make the WHOLE forward
bit-exact regardless of which survivor wins the subsequent speed race, because
bitwise equality is transitive. Measured, it is **not**: job 907292 (`x_bits.py`,
one process, `exact+noepi` vs `off`, `SALAD_OPT_DETSEG=1`, `base_rerun`
methodology control passing at exactly 0.0 so the harness itself is trusted):

| arm | pos max\|d\| | raw_angles max\|d\| | aatype flips |
|---|---|---|---|
| `all` (N020, unfiltered) | 5.84e-05 | 7.97e-03 | - |
| **`exact+noepi` (N030, per-shape verified)** | **5.44e-05** | 5.28e-03 | 0 |
| `exact+noepi+decomp32` | 3.72e-05 | 3.16e-03 | 0 |
| `decomp32` alone | 3.72e-05 | 3.16e-03 | 0 |
| `base_rerun` (methodology control) | **0.0 exactly** | 0.0 | 0 |

**Filtering to per-shape-verified-exact candidates removes only ~7% of the
deviation** (5.84e-05 -> 5.44e-05), not all of it. `exact+noepi+decomp32` is
bit-for-bit IDENTICAL to plain `decomp32` - i.e. once `decompose_small_bmm` is
on, the template filter contributes literally nothing more, good or bad; the
whole remaining deviation in that row is attributable to `decompose_small_bmm`
alone (section 5).

Whole-forward speed for the properly-filtered arm (`exact+noepi`, job
907226, `epilogue_fusion=False` so the verified kernel is what actually runs -
without `noepi` the arm silently matches `all`'s kernel counts exactly, job
907130, because `benchmark_epilogue_fusion` re-races a fused
template+epilogue that was never verified): **lean 3.3705 -> 2.9988 ms =
-11.0%** at L=100 (worse than `all`'s -14.0% AND worse than
`decompose_small_bmm`'s -15.5%/-16.6%, section 5) for a **657s -> 2500s**
compile time (~4x `all`'s own compile tax, ~8-9x plain `off`).

**Why per-call verification does not compose to a bit-exact whole**, to the
extent this pass could chase it down without turning into its own multi-day
investigation (out of scope for this item to fully resolve): the check
compares the Triton candidate against ONE explicit `aten[0].benchmark(...)`
call made by `_filter` at compile time, using inductor's own synthetic
`benchmark_example_value` inputs at that call site's real shape/stride. That
reference is provably correct FOR THAT ONE CALL in isolation. What it cannot
account for is inductor's subsequent, SEPARATE autotune benchmarking pass
(`orig(name, choices, ...)`, called right after `_filter` returns), which
re-benchmarks the same ATen candidate again as part of its own timing race
among the survivors - a second, independent invocation of the identical
kernel. Section 6 below (job 907764) shows this second-order effect is real
and not merely academic: it moves the bits between compiles, so whatever
subtle state or heuristic varies between two calls of "the same" cuBLAS
kernel invocation is exactly what breaks the "verify once, safe forever"
argument. Not fully isolated to a single root cause; flagged for whoever
revisits this rather than closed with a guess.

## 5. `decompose_small_bmm` (inductor's `decompose_mm_pass`) - the actual GEMM-merge win

`opt.py`'s `resolve_decompose_small_bmm` / `x_arm.py`'s `decomp<n>` feature,
answering the brief's item-2 ask ("GEMM merges that do not go through P7's
miscompiled code path") with something nobody had built yet: inductor SHIPS a
post-grad fusion pass, `decompose_mem_bound_mm.py`, that rewrites a batched
`bmm` with batch >= `min_first_dimension_decomposition` and two of (m, n, k)
under `max_other_dimention_decomposition` into
`(a[..., None] * b[..., None, :, :]).sum(-2)` - a Triton kernel instead of a
cuBLAS launch. Upstream's own default threshold (10240) misses every 3x3
frame-rotation `bmm` in this model (batch 100-18432, **but literally every one
has batch < 10240** at these lengths); asking for 32 catches ~90 of them.

**This is NOT an autotune race.** `decompose_mm_pass` is a static, threshold-
triggered FX-graph rewrite - no candidate benchmarking, no timing, no per-
device search. That already predicts it should be perfectly reproducible
across cold compiles, unlike `gemm_templates` in any mode; section 6 confirms
it.

### 5a. Speed - re-measured on the CURRENT baseline (P7 re-enabled, commit 40a85be+)

The item's first measurement (job 907130/907226, `RESULTS.md` section 2) was
taken on a tree with P7 (`batch_cond_pair`) still disabled - the shipped
default has since flipped to P7-ON (`40a85be`, "lean 3.074 ms"), and P7's own
`StackedCondPair` targets a batched condition-branch computation that COULD
have overlapped with `decompose_small_bmm`'s 3x3-frame-rotation target,
making the pre-P7 -16.6% number stale. Re-measured with P7 on (job 907765,
`x_bench.py`, real `FastDenoiser`, preset `exact` with `decompose_small_bmm`
toggled - see the harness-clobber fix in section 5c):

| arm | lean ms | vs off | full ms | kernels | triton | cuBLAS launches | cuBLAS us | GPU busy us | compile s |
|---|---|---|---|---|---|---|---|---|---|
| `off` (P7 on, shipped) | 3.0721 | - | 3.885 | 678 | 338 | 301 | 1870.9 | 2914.2 | 295 |
| `decomp32` (P7 on) | 2.5946 | **-15.5%** | 3.282 | 655 | 396 | 232 | 1297.8 | 2434.3 | 301 |

**-15.5% holds almost exactly** against the pre-P7 -16.6% (both measured the
same way, one arm per process, 3 reps within 0.1%) - P7 and
`decompose_small_bmm` are hitting different GEMMs and their savings are
close to additive, not competing. 69 fewer cuBLAS launches (301 -> 232), 573
us less cuBLAS time, GPU busy down 480 us (16.5%). Compile time is 295-301 s
either way - **no autotune tax**, unlike `gemm_templates` (657-2500 s).

### 5b. Bit-exactness - NOT exact, same order as before

Job 907766 (`x_bits.py`, one process, `off` vs `decomp32`, P7 on,
`SALAD_OPT_DETSEG=1`, `base_rerun` control at exactly 0.0):

**`decomp32` vs `off`: bit_identical=False, pos max\|d\|=4.96e-05 A**
(raw_angles 5.28e-03, aatype flips 0) - above the 2e-05 A floor, consistent
with the pre-P7 measurement of 3.72e-05 (same order of magnitude; P7 being on
shifts it slightly but does not change the verdict). **Cannot enter `exact` or
`deterministic`.**

### 5c. A harness bug this item's OWN new `FastDenoiser` wiring introduced, caught before it produced a false result

`x_bench.py` (used for 5a) builds a real `FastDenoiser(preset=...)`, which
after this item's changes resolves `decompose_small_bmm` from
`PRESETS[PRESET]` INSIDE `__init__` - a step that did not exist when
`x_bench.py` was first written and unconditionally OVERWROTE whatever
`x_arm.apply("decomp32")` had just set on `ind.post_grad_fusion_options`,
because `FastDenoiser` is constructed AFTER `x_arm.apply` in that harness and
the default preset's `decompose_small_bmm` is `False`. Caught via an added
assertion (`x_bench.py`, "FastDenoiser clobbered it") before trusting any
number from this section - exactly the "arm that looks on and is not" trap
this project's own notes warn about, this time self-inflicted by landing new
`FastDenoiser` logic without updating an existing harness that constructs one.
Fixed by also mutating `PRESETS[PRESET]["decompose_small_bmm"]` from the arm
string, mirroring how the harness already did it for `gemm_templates`.
`x_bits.py`/`x_dump.py` were NOT affected (they compile `forward_core`
directly, never touching `FastDenoiser`/`PRESETS`).

## 6. Can the autotune be PINNED? Three independent cold-cache compiles (job 907764)

The brief's core ask, answered directly: `off` (no autotune at all) as the
noise floor, `exact+noepi` (the per-shape bitwise filter) as the candidate
being tested for reproducibility, `decompose_small_bmm` as the non-autotune
comparison. One process per compile (`x_dump.py`), `SALAD_OPT_DETSEG=1`,
compared pairwise with `x_cmp.py`.

| arm | rep1 vs rep2 | rep1 vs rep3 |
|---|---|---|
| `off` (noise floor) | `pos`/`trajectory`/etc bitwise 0.0; `aa`/`aa_features` ~1e-6 (fp32-epsilon eager/compiled path noise, unrelated to any of this item's changes) | same, ~1e-6 |
| **`exact+noepi` (N030 filter)** | **bit_identical=True (0.0 everywhere)** | **bit_identical=False, pos max\|d\|=3.46e-05** |
| **`decompose_small_bmm=32`** | **bit_identical=True (0.0 everywhere)** | **bit_identical=True (0.0 everywhere)** |

**No, the autotune cannot be pinned by per-shape bitwise verification alone.**
`exact+noepi` agrees with itself on one pair of cold compiles and DISAGREES on
another, by an amount (3.46e-05) an order of magnitude above the `off` noise
floor (~1e-6) and in the same band as N020's original, unfiltered timing-race
finding (up to 8.87e-05). This means the instability is not confined to
"which already-verified candidate wins the speed race" (harmless, since all
survivors are bitwise equal to each other by construction) - **the SET of
candidates that survive verification is itself not stable across cold-cache
compiles.** `gemm_exact.py`'s `mode="stable"` (deterministic tie-break via
`hash_key()`) was built for exactly the ORIGINAL tie-break race and would fix
THAT, but does not touch this second, deeper instability and was not
separately re-tested here since it was already out of contention on
exactness grounds (`stable` never claimed bitwise equality to cuBLAS as a
goal - see its own docstring in `gemm_exact.py`).

`decompose_small_bmm`, having no autotune step at all, is **perfectly
reproducible** across all 3 cold compiles - the one candidate in this item
that is both fast (5a) and self-consistent (this section), just not bit-exact
against cuBLAS (5b).

## 7. Design-quality gate (n=160) for the ACTUAL shipped `fast` preset

Two separate n=160 runs, both `--preset fast`, both `diverged: 0/160`:

* Job 907298 (pre-final-wiring, `SALAD_OPT_DECOMPBMM=32` env override on the
  as-shipped `fast` preset, P7 still off at that point): CA-CA 3.7612 A,
  Rg 13.02, vs the SAME job's `baseline` arm (no decompose) at 3.7625 A /
  13.04 - indistinguishable.
* Job 907763 (post-final-wiring, `--preset fast` with **no env override at
  all** - i.e. the code path a real user hits, `decompose_small_bmm=32` now
  the preset's own default, P7 on): CA-CA 3.7638 A, Rg 12.96, `diverged: 0/160`.
* Job 907993 (final pre-commit re-run, section 9, on top of N028's O6
  `pair_feat_merge` landing too): CA-CA 3.7620 A, Rg 12.97, `diverged: 1/160 =
  0.6%` (limit 10%) - a single-design outlier, comfortably inside the pass
  threshold and consistent with ordinary run-to-run noise on the `tf32=True`
  `fast` preset (not a new effect of this item's change; `test_design_quality`
  is a statistical gate at n=160, not a bit-exactness check).

**`decompose_small_bmm=32` in `fast`: shippable per this project's own
decision rule** (N020/N015's rule: not bit-exact -> never `exact`, but a
reassociation that clears the n=160 structural gate IS a legitimate `fast`
feature).

## 8. Portability

* **`gemm_exact.py`** (both `exact` and `stable` modes): re-verifies bitwise
  equality (or re-times candidates for `stable`) at COMPILE time, on whatever
  device is present, naming no architecture, block size, warp count or stage
  count anywhere in the file (confirmed by inspection - the only literals are
  `32` as a default `max_other_dimention_decomposition`-style knob and
  generic Python). On an unseen card (a 4090, or Blackwell sm_120 once
  torch/Triton support it): every verification re-runs from scratch; a shape
  where no Triton candidate is bitwise-exact keeps today's cuBLAS kernel
  (`aten_only_by_K`/`untouched`/all-candidates-fail all degrade to the ATen
  choice, per `_filter`'s fail-safe). This is now MOOT for shipping purposes
  since section 4/6 close the door on `exact`/`stable` regardless of
  portability - kept only as a research knob, so its portability story is
  informational, not a shipping argument.
* **`decompose_small_bmm` (`decompose_mm_pass`)**: a pure shape/threshold
  rule (`batch >= min_first_dimension_decomposition and other dims < 32`),
  with NO autotuning, NO benchmarking and NO device query anywhere in
  inductor's own pass (confirmed by reading
  `torch._inductor.fx_passes.decompose_mem_bound_mm`). The SAME rewrite fires
  on any device for the SAME model shapes (L, batch counts and the 3x3 frame
  dimensions are architecture constants, not device-dependent), so an unseen
  card gets the IDENTICAL graph transformation VERBATIM - there is no
  per-device adaptivity here at all, for better or worse. Whether it is a NET
  win on that card is a question about relative cuBLAS-launch-latency vs
  Triton-kernel-floor cost, not measured on any card but this H100
  (torch 2.5.1+cu121 cannot target sm_120/Blackwell at all - same
  environment ceiling N020 hit). N017's own portability argument (section 6
  of that item) applies here unchanged: a consumer card's lower memory
  bandwidth relative to its fp32 rate makes the LAUNCH side of the trade
  (removing 69 cuBLAS launches at ~6us floor each) relatively MORE valuable,
  not less, so there is no reason to expect this to regress on a smaller
  card - but that is an argument from the shape of the trade, not a
  measurement, exactly as flagged for `decompose_small_bmm`'s upstream
  cousin `frame_mulsum` (N028/O8) in that item's own notes.

## 9. Rebasing onto concurrent HEAD moves and landing

This item's tree moved TWICE while this pass was running:
`98bc61b` (P7 off, this item's original snapshot) -> `40a85be` (N026
re-enabled P7) -> `0915f5a` (N028 landed O6, `pair_feat_merge`, bit-exact,
on every preset) - plus N028's OWN further uncommitted work-in-progress
(`frame_mulsum`/`ipa_mulsum`, targeting the SAME 3x3-frame-rotation family
`decompose_small_bmm` does, by a different mechanism - a source-level
`einsum` rewrite rather than a post-grad graph pass) sitting in the live
working tree throughout. Two jobs (907718 the 3-cold-compile determinism run,
907722/907723 the first ship-quality/gate attempt) were caught mid-race
against this item's own worktree edits and discarded/rerun clean (907718
cancelled; 907722/907723's md5sums did not match the final code and were
superseded by 907763/907762 and finally 907992/907993).

To commit ONLY this item's own diff without capturing N028's uncommitted
work: `git stash push` on the 6 files N028 had modified in the live tree,
confirmed the working tree then matched `0915f5a` exactly (`diff -q` against
`git show 0915f5a:...`), applied this item's changes as a standalone patch
generated against `0915f5a` (verified `git apply --check` clean), ran the
FINAL pre-commit gate on that exact state (this section), committed, and
`git stash pop` to restore N028's WIP on top afterward. Not a rebase or
merge - `decompose_small_bmm`'s and `frame_mulsum`/`ipa_mulsum`'s edits are
in disjoint regions of the same two files (confirmed by diff before
stashing), so the stash pop is a clean line-level union, not a 3-way merge
of overlapping logic. **Composition with `frame_mulsum` once it lands is
UNMEASURED** - both target the same ~90 3x3-frame `bmm`s by different
mechanisms (`frame_mulsum` rewrites the SOURCE `einsum` calls before ATen
ever sees a `bmm`; `decompose_small_bmm` intercepts the `bmm` AFTER ATen
lowering), so once `frame_mulsum` lands, `decompose_small_bmm`'s effective
target set will likely SHRINK (there may be nothing left for it to
decompose in that family) rather than compound - flagged for whoever lands
`frame_mulsum` next, not measured here.

**Final pre-commit gate** (`salad_jobs/n030_final_cpu.sbatch` job 907992,
`salad_jobs/n030_final_gpu.sbatch` job 907993), run on `0915f5a` + this
item's changes ONLY (N028's WIP stashed away):

* CPU (907992): 4/4 `test_model_parity` configs `EXIT 0`, `test_geometry_parity`
  `EXIT 0`, `test_stack_parity`/`test_encoder_parity`/`test_sampling_parity`
  `EXIT 0`. **`CPU_SUITE_EXIT=0`.**
* GPU (907993): same 8 tests `EXIT 0`, plus `quality fast n=160` `EXIT 0`
  (1/160 diverged, 0.6%, well under the 10% limit - see section 7), `quality
  exact n=64` `EXIT 0` (0/64 diverged - `exact` is untouched by this item's
  default, as expected), `group_size auto` (`scripts/generate.py` sanity)
  `EXIT 0`. **`GPU_SUITE_EXIT=0`.**

## 10. Decision

Per the brief's own fallback ("If neither works, say so with the numbers and
close the item"):

* **Pinning the autotune (brief bullet 1): NO.** Even the bitwise-verified
  candidate filter is not reproducible against itself across cold compiles
  (section 6) - a strictly harder failure than the tie-break race it was
  built to fix, since the CANDIDATE SET, not just the winner among it, moves.
* **Per-shape allowlist / bitwise filter (brief bullet 2): NO.** Verified,
  measured, NOT bit-exact end to end (section 4: 5.44e-05 A, barely better
  than unfiltered) and it is SLOWER (-11.0%) than the alternative this item
  found (section 5) for many times the compile cost.
* **`gemm_templates` in any mode (`all`/`exact`/`stable`) do NOT ship in any
  preset.** Kept as opt-in research/debug knobs only (`SALAD_OPT_GEMMTPL=
  all|exact|stable`), matching N020's original disposition.
* **`decompose_small_bmm=32` SHIPS in `fast` ONLY** (`False` in `exact` and
  `deterministic`, matching the project's standing rule that a reassociation
  needs the preset's own numbers to move before it earns a spot, never the
  bit-exact presets): -15.5% at L=100 on the current P7-on baseline (largest
  cuBLAS-side number since P7 itself), no autotune tax (300s compile, same as
  `off`), perfectly reproducible across cold compiles, and now cleared BOTH
  gates on the actual shipped preset code (section 9).

Concretely, in the code:

* `salad_torch/gemm_exact.py` (NEW) - the verification/stable-pick module,
  unused by any preset, importable and usable via `SALAD_OPT_GEMMTPL=exact`
  or `=stable` for future research only.
* `salad_torch/opt.py` - `GEMM_TPL_MODE`/`resolve_gemm_templates` (mode
  resolution, replacing the old boolean `GEMM_TEMPLATES` semantics while
  keeping it as a back-compat alias) and `DECOMPOSE_SMALL_BMM`/
  `resolve_decompose_small_bmm` (the `SALAD_OPT_DECOMPBMM` threshold/veto).
* `salad_torch/modules/fast.py` - `PRESETS["fast"]["decompose_small_bmm"] =
  32` (the only preset-default CHANGE this item makes); `exact`/
  `deterministic` both explicitly set `decompose_small_bmm=False`;
  `FastDenoiser.__init__` wires both `gemm_exact.install/uninstall` (salted
  into `joint_custom_post_pass`'s NAME, not just its config, so the FX-graph
  cache cannot serve one mode's compiled artifact to another) and
  `decompose_mm_pass`'s `post_grad_fusion_options`.

## Next

* `frame_mulsum`/`ipa_mulsum` (N028) target the same 3x3-frame `bmm` family
  as `decompose_small_bmm` by a different mechanism (source rewrite vs
  post-grad pass) - re-measure `decompose_small_bmm`'s marginal contribution
  once `frame_mulsum` lands; it will very likely shrink, not stack.
* The mechanism behind section 4/6's instability (per-call verification
  passing while the candidate SET itself drifts across cold compiles) is not
  isolated to a specific torch/cuBLAS internal - flagged, not chased, given
  this item's scope was "can it be shipped", answered NO either way.
* `gemm_exact.py`'s `mode="stable"` was built, never separately validated at
  n=160 (moot for `exact`/`deterministic` regardless, and slower than
  `decompose_small_bmm` for `fast`) - if someone wants deterministic-but-not-
  cuBLAS-exact templates for some OTHER reason later, the code exists.
