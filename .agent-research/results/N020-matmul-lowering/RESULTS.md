# N020 - HkLinear matmul lowering: batch-1 `bmm` -> `mm`, and inductor's matmul templates

Tree: `n006-manual-cudagraph` @ fd9ba32 + this item. Work snapshot
`/hnvme/workspace/v181bb10-salad/n020` (a plain copy, not a git worktree, so it
costs ~130 inodes and cannot disturb N021/N022 in the shared tree). The
previous agent on this item died on a rate limit after producing most of the
data below (everything under `cpu_*`, `gate_904335.log`, `gpu_bench*`,
`gpu_bits_904338.log`, `gpu_shapes_904337.log`, `gpu_stability_904348.log`,
the sbatch scripts and `x_*.py`); this agent verified that work, filled the
one gap it left (deliverable 1's `einsum`+templates bitwise control was never
run - only `mm`/`addmm`+templates were), fixed two now-falsified code comments
and a wrong default it had shipped, and closed the item.

## 0. The change

`salad_torch/modules/basic.py:73`, every linear layer in the model:

```python
-        out = torch.einsum("...k,ki->...i", x, self.w)      # -> aten.bmm, batch=1
-        if self.use_bias:
-            out = out + self.b
+        mode = opt.MM_MODE
+        if mode == "einsum":
+            out = torch.einsum("...k,ki->...i", x, self.w); out = out + self.b if self.use_bias else out
+            return out
+        flat = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
+        out = torch.addmm(self.b, flat, self.w) if (mode == "addmm" and self.use_bias) else torch.mm(flat, self.w)
+        ... + bias, + reshape back when ndim != 2
```

The Haiku `(in, out)` weight layout is untouched (no transpose anywhere), so
checkpoint conversion stays a pure name mapping. `forward_indexed` is
untouched. The unused `b` parameters are still allocated and still never read
when `bias=False` (202 of them; DDP's `find_unused_parameters=True` depends on
it).

Two flags, both following the project's veto convention (`salad_torch/opt.py`):

* `SALAD_OPT_MM=einsum|mm|addmm` - **`einsum` is the default** (fixed by this
  agent; the snapshot handed off had defaulted to `mm`, which is not
  bit-exact and must never be the default - see section 2). `mm`/`addmm`
  remain available as bisection/debug knobs only; neither ships anywhere.
* `SALAD_OPT_GEMMTPL=0|1` - inductor's `max_autotune_gemm` (matmul templates).
  **Default `False`, and no preset requests it** (fixed by this agent; the
  snapshot handed off had it wired into all three presets in `modules/fast.py`
  next to `aggressive_fusion`, before the bitwise control below had been run).

**`addmm` was tried and is REJECTED** - see section 2. It is not bit-exact
against the einsum form, and it FAILS the parity gate outright on
`multimotif_vp`.

## 1. Recap: what was already established before this agent (previous agent + brief)

Measured at L=100, `exact`-equivalent settings, lean step, one length per
process (`gpu_bench.sbatch` / `gpu_bench2.sbatch`, jobs 904339/904341,
re-confirmed by this agent - see section 5 for the full table with L=400
added):

| arm | mm_mode | templates | lean | kernels | cuBLAS | Triton | templates |
|---|---|---|---|---|---|---|---|
| base | einsum | off | 3.071 ms | 677 | 301 / 1879 us | 338 / 1004 us | 0 |
| mm | mm | off | 3.063 ms | 670 | 301 / 1880 us | 331 / 987 us | 0 |
| **einsumtpl** | **einsum** | **on** | **2.643 ms** | **559** | **59 / 391 us** | 286 / 924 us | **212 / 1178 us** |

Templates replace 242 cuBLAS launches with 212 Triton template kernels, for
-14.0% of a lean step. The `einsum -> mm` lowering on its own buys -0.3% and
does not reduce the cuBLAS launch count at all (301 -> 301) - **the premise
that batch-1 `bmm` forecloses inductor's matmul templates is WRONG**: inductor
has its own `tuned_bmm` template path, so `einsumtpl` (einsum completely
unchanged) gets the whole -14% without the `mm` rewrite. This corrects both an
earlier brief and N017's inference, and this agent propagated the correction
into `opt.py`'s and `fast.py`'s docstrings (they had claimed templates were
"only reachable because `HkLinear` now lowers to `aten.mm`", which is false).

## 2. `einsum -> mm`/`addmm` lowering: REJECTED

CPU bitwise control (`cpu.sbatch`/`cpu_diag.sbatch`/`cpu_eager.sbatch`, jobs
904334/904356/904374), compiled `forward_core`, all eleven outputs:

| L | mm vs einsum, pos max\|d\| | addmm vs einsum, pos max\|d\| |
|---|---|---|
| 64  | 5.579e-05 | 5.579e-05 |
| 100 | 4.196e-05 | 4.196e-05 |

Both are **ABOVE the 2e-05 A parity floor** (`local` and `raw_angles` are
worse still: up to 2.6e-03 / 3.0e-03). `cpu_diag_904356.log`'s isolated-knob
sweep (CSE off/on x aggressive_fusion off/on, same numbers in all four
combinations) shows the delta is not from either fusion pass - it is inductor
choosing a different summation order for the SAME arithmetic once the op is
`aten.mm` instead of `aten.bmm`. Eager `mm` vs eager `einsum` is exactly
bit-identical (`cpu_eager_904374.log`), so nothing at the python level moved -
only inductor's codegen for the two different ops did.

The full parity gate (`gate.sbatch`, job 904335) confirms the consequence:

* `SALAD_OPT_MM=mm` passes the coarse Angstrom-floor parity suite anyway
  (`SUITE_EXIT=0`) - the suite is eager/CPU only (see section 3's caveat) and
  its floor happens to be loose enough on this particular delta.
* `SALAD_OPT_MM=addmm` **FAILS outright** on `multimotif_vp`:
  `local rel=4.451e-05 abs=3.94e-01 (FAIL)`, plus `aa`, `raw_angles`, `dssp`,
  `prev_local` all FAIL too.

So `mm`/`addmm` buy almost nothing (-0.3%, no launch-count reduction), aren't
bit-exact, and one of the two fails the standard gate outright. **Rejected.**
Kept as `SALAD_OPT_MM=mm|addmm` debug/bisection knobs; default fixed to
`einsum` in this pass (see section 0).

## 3. Deliverable 1: are the matmul templates bit-exact end to end?

### 3a. CPU control (`cpu_tplcontrol.sbatch`, job 904402)

`SALAD_OPT_MM=einsum` (the shipped default) throughout, `SALAD_OPT_GEMMTPL=0`
vs `=1`, all eleven `forward_core` outputs:

| L | bit_identical |
|---|---|
| 64  | **True** (max\|d\| = 0.0 exactly, every output) |
| 100 | **True** (max\|d\| = 0.0 exactly, every output) |

**Templates do not engage on CPU at all** - `max_autotune_gemm`'s Triton
template backend is CUDA-only in this inductor version, so this control is a
trivial pass and proves nothing about the GPU kernel selection, exactly as
anticipated. Falling back to the GPU control, per the task brief.

### 3b. GPU control, `SALAD_OPT_DETSEG=1` (`gpu_tplcontrol.sbatch`, job 904395)

The one control the previous agent's jobs never ran: `mm_tpl1`/`addmm_tpl1`
were checked bitwise (`gpu_bits.sbatch`, job 904338) but not `einsum_tpl1` -
the combination that actually ships now that `SALAD_OPT_MM` defaults to
`einsum`. Same node, same job, tpl0 then tpl1 compiled back to back, all
eleven `forward_core` outputs:

| L | bit_identical | pos max\|d\| | local | raw_angles |
|---|---|---|---|---|
| 100 | **False** | 5.841e-05 | 4.227e-03 | 7.965e-03 |
| 400 | **False** | 7.870e-04 | 2.037e-02 | 1.037e-01 |

**Not bit-exact.** `pos` at 5.84e-05 is 2.9x the 2e-05 A parity floor at
L=100, and the gap widens with length. **Required `max|d|=0.0` exactly for a
bit-exactness claim; got neither.**

### 3c. A confound worth stating plainly: DETSEG's own cross-compile noise

The prior agent's `gpu_bits.sbatch` (job 904338) included a same-config rerun
arm (`einsum_tpl0_rerun` vs the original `einsum_tpl0`, **zero code or flag
change, just an independent cold-cache recompile**):

| L | rerun vs original, pos max\|d\| | note |
|---|---|---|
| 100 | 3.457e-05 | same order of magnitude as the templates delta above |
| 400 | 1.367e-03 | **larger** than this item's own templates delta (7.87e-04) at the same L, and flips 1 discrete `aatype` residue |

So under `SALAD_OPT_DETSEG=1`, an independent cold-cache recompile of the
*identical* configuration is itself not bit-identical to the original at
these lengths (likely PYTHONHASHSEED-driven scheduling/codegen-order variance
in inductor across process boundaries - not investigated further, out of
scope for this item, flagged for a future one). This means the GPU-DETSEG
methodology cannot cleanly attribute "how much of the delta is templates
specifically" once two independent compiles are involved - **but it does not
change the answer**: no comparison in this section - templates-attributed or
plain-recompile-noise - was ever `0.0`, and the isolated same-job test in 3b
(the cleanest available) is unambiguous on its own: 5.84e-05 at L=100 is
templates-attributable on its own terms (both arms compiled in the same job,
back to back) and already fails the floor before any confound is considered.

### 3d. Reproducibility of the autotune choice itself (`gpu_stability.sbatch`, job 904348)

Three independent cold-cache compiles of the SAME `mm`+templates-on
configuration, compared bitwise under DETSEG:

| pair | pos max\|d\| |
|---|---|
| r1 vs r2 | 4.768e-07 (aa only; effectively fp32 epsilon noise) |
| r1 vs r3 | 8.869e-05 |
| r1 vs templates-off | 1.268e-04 |

`max_autotune_gemm`'s selection is a **timing race**: r1/r2 agree to ~4 ULP,
but r3 disagrees by an amount comparable to the templates-on-vs-off delta
itself. At a shape where cuBLAS and a Triton template are within
benchmarking noise of each other, independent compiles can and do pick
different winners - i.e. the templates config is not even reproducible
against **itself**, let alone against templates-off. This independently rules
out "exact" (whose whole model is "one specific set of bits, forever") even
before the parity-floor argument in 3b.

**Verdict: NOT bit-exact.** The GPU DETSEG control (3b) is the one this task
asked for and it fails on its own; 3c and 3d only reinforce it from two
different angles.

## 4. Deliverable 2: standard parity suite with `SALAD_OPT_GEMMTPL=1`

Two runs, both `SUITE_EXIT=0`:

* `gate_tpl.sbatch` (job 904396), mid-edit code: `SALAD_OPT_MM=einsum
  SALAD_OPT_GEMMTPL=1` explicit -> all four configs (`default_vp`,
  `default_vp_scaled`, `default_ve_scaled` at floor 2e-05 A;
  `multimotif_vp` at 2e-04 A) PASS, `test_geometry_parity` 38/38,
  `test_stack_parity`/`test_encoder_parity`/`test_sampling_parity` all OK.
* Re-run on the FINAL code after all fixes in this pass (job 904646, arm
  `explicit_einsum_gemmtpl1_final`) and the as-shipped default with no env
  vars at all (job 904649, arm `ship_default`): see section 8 for the final
  confirmed exit codes.

**Necessary but not sufficient caveat, stated explicitly per the task's own
framing:** every one of these tests (`test_model_parity.py`,
`test_geometry_parity.py`, `test_stack_parity.py`, `test_encoder_parity.py`,
`test_sampling_parity.py`) runs the **eager, uncompiled** model on CPU
(confirmed by grep - none of the five files reference `torch.compile`,
`FastDenoiser`, or any inductor config). `SALAD_OPT_GEMMTPL` only has an
effect inside `torch._inductor`, which this suite never invokes. So "the
suite passes with `SALAD_OPT_GEMMTPL=1`" is real, honestly reported evidence
that setting the flag does not regress anything reachable from eager CPU
code - but it is structurally blind to the one thing this whole item is about
(the GPU kernel selection), exactly as `fast.py`'s own N017 comment already
warns ("the four checkpoint parity tests run the EAGER path and cannot see a
compile-time graph change at all"). Section 3's bitwise GPU control is the
evidence that actually bears on the decision.

## 5. Deliverable 3: does -14% hold at L=400?

`gpu_bench.sbatch` (job 904339, arms `base`/`mm`/`mmtpl`/`addmmtpl`) +
`gpu_bench2.sbatch` (job 904341, arm `einsumtpl`), 3 reps each, one
(arm, length) per process:

| arm | L | lean (min/mean, ms) | kernels | cuBLAS | Triton | templates | compile |
|---|---|---|---|---|---|---|---|
| base | 100 | 3.071 / 3.072 | 677 | 301 / 1879us | 338 / 1004us | 0 | 298s |
| mm | 100 | 3.063 / 3.063 | 670 | 301 / 1880us | 331 / 987us | 0 | 286s |
| mmtpl | 100 | 2.606 / 2.606 | 558 | 59 / 388us | 285 / 915us | 212 / 1154us | 697s |
| **einsumtpl** | **100** | **2.643 / 2.644** | 559 | 59 / 391us | 286 / 924us | 212 / 1178us | 740s |
| base | 400 | 7.078 / 7.079 | 654 | 278 / 4035us | 338 / 2861us | 0 | 390s |
| mm | 400 | 7.063 / 7.063 | 647 | 278 / 4044us | 331 / 2845us | 0 | 382s |
| mmtpl | 400 | 6.589 / 6.590 | 553 | 74 / 1439us | 279 / 2725us | 198 / 2307us | 787s |
| **einsumtpl** | **400** | **6.624 / 6.625** | 560 | 80 / 1545us | 280 / 2731us | 192 / 2203us | 838s |

`einsumtpl` (the arm that actually ships, since `SALAD_OPT_MM` stays `einsum`):

* **L=100: 3.071 -> 2.643 ms = -13.9%** (confirms the established number).
* **L=400: 7.079 -> 6.625 ms = -6.4%** (NEW - not previously measured). **The
  win does not hold at length.** Same failure mode as P7 (N014): -9.2%/-11.0%
  at L=100 eroding to -2.3%/-2.3% at L=400, though templates erode less
  severely (~2.2x shrink vs P7's ~4x).
* Mechanism: the number of narrow, fixed-shape GEMMs templates win on does
  not scale with L (weight shapes are architecture constants), so the
  absolute time saved is roughly constant (~470us busy-time at L=100, ~470us
  at L=400: GPU busy 2928->2459us and 6955->6489us respectively) while the
  rest of the step scales with L, diluting the percentage.
* Compile time: 298s (no templates) -> 740-838s (templates on), i.e. ~7-9x,
  consistent with the "~12 min" budget in the brief. One-time cost, as
  expected.
* 3 reps per arm, all within 0.1% of each other (see raw arrays in the job
  logs) - stable, not noise.

## 6. Portability (hard requirement from the user)

* `grep -rn "BLOCK_SIZE\|num_warps\|num_stages\|search_space\|autotune_fallback"
  salad_torch/` - **zero hits** outside this docstring. Nothing pins a block
  size, warp count or stage count anywhere in the shipped code; every
  `BLOCK_M/N/K, num_warps, num_stages` combination visible in the autotune
  logs (`gpu_shapes_904337.log`, `gpu_bench_904339.log`) comes from inductor's
  own candidate search, not from this change.
* `max_autotune_gemm_backends = "ATEN,TRITON"` keeps ATEN in the candidate
  list, so a shape where every Triton candidate loses (or where none even
  compiles) keeps exactly today's cuBLAS kernel. Observed directly in the
  logs: several shapes throw `Exception out of resource: shared memory,
  Required: 262144, Hardware limit: 232448` for specific `BLOCK_*`/`num_stages`
  candidates - inductor logs the failure, drops that candidate, and picks
  from the survivors (or falls back to ATEN if none survive). This is the
  graceful-degradation path working as intended, observed live, not merely
  assumed.
* Version-sensitive assumption, flagged rather than tested per the brief:
  this environment is torch 2.5.1+cu121, which cannot target sm_120
  (Blackwell) at all, so none of the above was exercised on a Blackwell or
  consumer-Ada card. The mechanism (autotune benchmarks per shape at compile
  time on whatever device is present) is architecturally card-agnostic, but
  that is an argument from design, not a measurement, on this hardware/torch
  combination.

## 7. Decision

Per the task's decision rule:

> suite passes but NOT bit-exact -> it does NOT go in `exact`. It may go in
> `fast`, but only with an n=160 structural validation first.

**Section 3 found NOT bit-exact** (pos max|d| 5.84e-05 @ L=100 / 7.87e-04 @
L=400, both above the 2e-05 A floor, plus the config is not even reproducible
against itself across cold-cache compiles - section 3d). **Section 4 found
the suite passes**, with the stated caveat that the suite cannot see this
change at all.

**-> Templates do NOT land in `exact`.** They also do NOT land in `fast`
in this pass: no n=160 structural validation has been run (that is a
separate, ~day-scale experiment matching N015's method - generate ~160
designs under `fast`+templates vs `fast` alone and compare SSE/Rg/clash
distributions), and this project's own record (N011's n=8, N015's own
description of n=8/n=40 "producing confident wrong answers twice") means
shipping a numeric-perturbing 6-14% win into a default preset without that
validation is exactly the mistake this project has already made and
corrected for twice. **Landing nothing beyond an opt-in debug flag is the
correct, conservative close for this item.**

Concretely, in the code (all in the snapshot, ported to the live tree in the
commit for this item):

* `SALAD_OPT_MM` defaults to `einsum` (fixed - was wrongly `mm`).
  `mm`/`addmm` remain as rejected, opt-in-only bisection knobs.
* `SALAD_OPT_GEMMTPL` defaults to `False` (fixed - was `True`), and **no
  preset requests it** (`exact`, `fast`, `deterministic` all carry
  `gemm_templates=False` explicitly, with a comment pointing at this
  file, rather than silently dropping the key).
* Every place that had claimed "mm is bit-identical to einsum" or "templates
  are only reachable via the mm rewrite" - both false, per sections 2 and 1 -
  is corrected in `opt.py`'s and `fast.py`'s docstrings.

If a future item wants to pursue `fast`+templates (a real 6-14% win, just
not proven safe to ship by itself), the path is: n=160 structural validation
per N015's method, then land `gemm_templates=True` in the `fast` preset only
(never `exact`/`deterministic`, which both require bit-exactness this change
does not have).

## 8. Final gate re-run on the fully-fixed code

Two more CPU jobs, on the code state as actually committed (md5sums:
`basic.py e0093fd8...`, `opt.py 284cb11b...`, `fast.py af2c5782...`):

* Job 904649, arm `ship_default` - **no env vars set at all**. Confirmed in
  the log itself: `MM_MODE= einsum GEMM_TEMPLATES= False`. All four
  `model_parity` configs `exit=0`, `test_geometry_parity` 38/38, remaining
  three tests `exit=0`. **`SUITE_EXIT=0`.**
* Job 904646, arm `explicit_einsum_gemmtpl1_final` -
  `SALAD_OPT_MM=einsum SALAD_OPT_GEMMTPL=1` explicit, same fully-fixed code.
  Same results: all four configs `exit=0`, geometry 38/38, remaining three
  `exit=0`. **`SUITE_EXIT=0`.**

Both confirm the as-shipped default (`einsum`, templates off) and the
explicit "what if someone turns templates on anyway" arm pass the full suite
on the exact code being committed.

**Final check on the actual LIVE git tree** (job 904659, after porting the
snapshot's validated `basic.py`/`opt.py` wholesale and hand-applying only the
`fast.py` hunks that don't touch N021's concurrent `batch_size` work in the
same file - see section 10): `SUITE_EXIT=0`, all four configs, geometry
38/38, stack/encoder/sampling OK, on top of live HEAD `7eb9a10` (which by the
time of this check already included N021's staged batch-size changes AND an
independent fix for the `_cse_graph_pass` unpicklable-closure bug this agent
had also noticed independently in the snapshot's compile logs - see section
10). `basic.py`/`opt.py` md5sums match the validated snapshot exactly.

## 9. Cleanup

`/hnvme` inode grace: all `/tmp/n020*` scratch dumps and
`TORCHINDUCTOR_CACHE_DIR`s removed by each sbatch script on exit. The
snapshot at `/hnvme/workspace/v181bb10-salad/n020` stays (per the task's
"deliver from your snapshot" guidance) until explicitly told to remove it;
it is a plain copy (~130 inodes), not a worktree.

## 10. Porting the snapshot to the live tree

`git status` at the start of this pass showed `salad_torch/modules/fast.py`,
`salad_torch/diffusion/sampling.py` and `scripts/generate.py` already STAGED
with N021's uncommitted `batch_size`/packed-batching work (N021 owns the
sampler and grouping/batching code in `fast.py`, per this item's territory
split) - `basic.py` and `opt.py` were untouched by anyone else in the live
tree. So:

* `basic.py`, `opt.py`: copied the validated snapshot files wholesale (no
  conflicts possible - confirmed via `git diff HEAD` showing them unmodified
  before the copy).
* `fast.py`: `git diff <(git show HEAD:...) <snapshot>` showed the hunks are
  disjoint from N021's (PRESETS dict entries + the `__init__` inductor-config
  block vs N021's `batch_size`/`GroupFastDenoiser`/`_blank_inputs` changes),
  so each of the four hunks (three preset `gemm_templates=False,` lines + the
  `max_autotune_gemm` block) was hand-applied with the `Edit` tool against the
  CURRENT live file, not `git apply`d wholesale - `git apply` would have
  silently reverted an UNRELATED fix that landed on this branch while this
  agent was working (see below).
* Bonus finding, not this item's to fix: the snapshot's `fast.py` still had
  the OLD, buggy closure-based `_cse_graph_pass` (N017/O1) - the one that logs
  `Can't pickle local object 'FastDenoiser.__init__.<locals>._cse_pass'` and
  silently disables inductor's on-disk FX-graph cache (visible in this item's
  own `gpu_bench_904339.log`, arm `addmmtpl_L100`). Commit `7eb9a10` ("fix:
  O1's CSE pass was an unpicklable closure...") landed on this branch,
  independently, by another agent, WHILE this item was running, fixing
  exactly that. Porting `basic.py`/`opt.py` wholesale and `fast.py` by
  surgical hunks (rather than overwriting `fast.py` wholesale from the stale
  snapshot) preserved that fix instead of reverting it.
* Final check: full parity suite re-run directly on the live tree post-port
  (job 904659, `SUITE_EXIT=0`) - see section 8.

## Next

* An n=160 structural validation of `fast`+`gemm_templates=True` (N015's
  method) would let the item be re-opened to land it in `fast` specifically -
  not attempted here, out of scope for closing N020.
* Section 3c's finding - that `SALAD_OPT_DETSEG=1` does not actually
  guarantee bit-identical output across independent cold-cache recompiles at
  L=400 (a discrete `aatype` flip on a pure no-op rerun) - was not chased
  further (out of scope for this item) but potentially affects how much
  weight N013/N019's own "bit-identical under DETSEG" claims can bear at
  length; worth a dedicated look if those presets' claims are relied on at
  L>100 in the future.
