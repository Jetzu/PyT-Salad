# N018 - numerical-relaxation levers for the `fast` preset: what actually moves the step

Measured 2026-09-26 on an H100, `default_vp-200k`. Runs from a clean **detached
worktree** at `/hnvme/workspace/v181bb10-salad/n018-worktree` (HEAD `a8533a7`,
deleted after this job to respect the inode grace period), not the shared
checkout: another agent had uncommitted edits to `blocks.py`, `diffusion.py`,
`fast.py`, `opt.py`, `segment.py` and a new `cond_prologue.py` the whole time
this ran (P7/"batch_cond_pair" and a "deterministic" preset are *mid-flight*
there, not yet in any commit this task builds on). Patch: `nbr_mode.patch`
(applies clean to `a8533a7`, `patch -p1 --dry-run` verified); full modified
files also included (`opt_modified.py`, `diffusion_modified.py`,
`fast_modified.py`) since the worktree itself no longer exists.

GPU cost: 5 jobs, **~1.1 GPU-hours** total (stated before submitting, all
individually far under the 25 GPU-h approval gate).

## 0. Verdict

**Neither new lever is recommended for the shipped `fast` preset.** Both
reduce the neighbourhood `forward_core` builds (K=64/48 down to K=32/48-40) on
top of `fast`'s TF32, and both make the per-step kernel work genuinely
smaller - this is real work reduction, not schedule shrinkage, confirmed at
n=160:

* **N011's K64->32 cut ("halfnbr") is confirmed REJECTED, and more decisively
  than N011 could show at n=8.** Clash count explodes: 2.33 vs `exact`'s 0.25
  per design, Welch t=+7.56, **p=0.000**, Brown-Forsythe spread W=45.79,
  **p<0.0001**. This is not a marginal, noise-level effect the way N011's own
  n=8 helix/Rg numbers turned out to be (see §1) - it is a large, massively
  significant one that gets WORSE with more data, not better.
* **The milder cut this task proposes ("randhalf" - halve only the random
  neighbour component, leave the spatial one untouched) avoids the clash
  catastrophe** (0.29 vs 0.25, p=0.592, clean) **but is not quality-neutral
  either**: Rg spread blows up (sd 6.49 vs `exact`'s 1.76, Brown-Forsythe
  **p=0.0015**, survives Bonferroni), the Rg distribution shape moves
  (two-sample KS **p=0.0013**), and helix/strand fraction shifts by
  +0.117/-0.105 (Welch **p=0.002**/**p=0.003**, both survive Bonferroni
  alpha=0.0042 for this 12-test family). 17/160 designs land outside `exact`'s
  own 3-sigma Rg band, against 2/160 for `exact` itself.

So the honest finding is: **shrinking the neighbourhood is the one lever in
this whole investigation that changes what the network sees rather than how
precisely it computes, and that is exactly why it is the only one that keeps
buying real per-step time - and the only one that keeps costing real
structure, at every cut size tried.** The speed number is genuine (see §2);
the quality bill is real too, and it does not go away by making the cut
milder, only smaller.

## 1. Assessment of N011's prior negative results

The brief asked specifically whether N011's rejections were *measured badly*
or *genuinely don't pay*. Read `RESULTS.md`, `x_preset.py`, `x_fast.py`,
`x_gen_k32.py` and every raw log in `.agent-research/results/N011-fast-preset/`
end to end. Verdict, item by item:

| N011 finding | my assessment |
|---|---|
| bf16 GEMM (`matmul_precision("medium")`) slower than TF32 (2.856 vs 2.760 ms) | **Well measured, and mechanistically sound.** Isolated single-forward comparison (`x_fast.py`), same harness as the TF32 number it's compared against, and it comes with an independent kernel census (321 cuBLAS launches @ 6.5 us average, i.e. launch-bound not math-bound) that explains *why* bf16 doesn't help: on these GEMM shapes the wall clock is dominated by launch latency, not arithmetic throughput, so narrowing the mantissa further buys nothing once TF32 has already taken the one throughput-bound projection. I did not re-run this - the reasoning and the isolation are both sound, and nothing in my own measurements contradicts the launch-bound diagnosis (see the K-lever's *bigger* payoff at L=400 in §2, which is exactly what a partly-launch-bound, partly-shape-bound forward should do as N grows). |
| combo_kernels / `topk` (§6, "the one lever left") | **Under-reported, not badly measured.** The job (903189) ran and the log (`logs/combo_903189.log`) has the number, but RESULTS.md §6 ends on "*(result below)*" with nothing following - the result was never transcribed. For the record: `combo_topk` was 2.978 ms lean against `base`'s 3.007 ms, i.e. **-1.0%**, and the pure `combo` variant (no `topk`) crashes Inductor codegen (`ops.sort is only supported in persistent reductions`). So the verdict ("doesn't pay") is right, but it was asserted, not shown, in the document that ships it. `topk` alone was independently re-confirmed slower (+5.6%, `speed1_903089.log`) in three separate contexts across N011's own logs, which is real replication, not chance. |
| K 64->32 both branches: -10% of the step, but coil 0.076->0.109 and clashes 0.1->1.1/design at **n=8** | **Speed: well measured** (isolated harness *and* the production preset agree to within noise: `x_fast.py`'s `both32` 3.048 ms vs the production `x_gen_k32.py` config; my own from-scratch reproduction of the identical (16,8,8)/(16,8,8) cut gets 2.714 ms lean at L=100 on top of TF32, consistent in direction and magnitude). **Quality: correctly flagged as insufficient, and the flag was right to worry** - N011's own §4 explicitly calls this "suggestive, not established" at n=8, and N015 elsewhere in this project independently proved n=8 and even n=40 are not enough to trust a directional SSE/Rg result (the n=40 Rg-spread finding that vanished at n=160). Re-running properly at n=160 (§3 below) does NOT reverse the verdict, though - it makes it much stronger. So this is a case of "correctly identified as unreliable evidence, and the underlying effect turned out to be real anyway," not "measured badly." |
| shared self-conditioning draw, cache-key bypass, `need_aux=False`, neighbour-reuse-by-stride | Not re-tested here (out of scope for a *numerical-relaxation* lever - these are reassociation/control-flow proposals that either changed nothing measurable or crashed Dynamo, and none of them touch precision or shape). N011's own process-note about the monkeypatch silently patching a dead function (§5.2) is a real methodological catch, correctly self-reported. |

**Overall: N011's methodology on the numbers it reported was sound.** The one
gap was that the K-reduction quality gate ran at n=8 - which N011 itself
flagged as insufficient - and the brief was right to ask for it to be redone
at adequate n before treating "does not pay" as settled. Redone at n=160, the
verdict does not flip: it is *more* clearly rejected, and the extra
information gained is that even the milder version of the same idea still
costs something real, just a different something.

## 2. Speed: neighbourhood reduction is the one lever that pays MORE at bigger L

Production `FastDenoiser(preset=...)` (not a monkeypatch), `bench_speed.py`
(the harness compiles it captures the whole forward with CUDA graphs and
replays it; `fast()` = reporting/"full" step, `fast(need_aa=False)` = lean
step). `steps_scale=1.0` for every config here - **the schedule was never
touched**; only the compiled kernel work changes between rows, per the
brief's explicit requirement. One process per (length, config) - K changes
tensor shape, so `exact`/`fast` (K=64, safe together) and each of the two new
K-configs (K=32-ish, K=48/40) each got their own process. 3 reps per config,
sd shown; run-to-run noise is <0.02 ms throughout, well below any of the
effects below.

| preset | L | nbr shape (cur / cond) | lean step | vs `exact` | vs `fast` (TF32-only) |
|---|---|---|---|---|---|
| exact | 100 | 16,16,32 / 16,16,16 | 3.572 ms | - | - |
| fast (TF32 only) | 100 | 16,16,32 / 16,16,16 | 3.007 ms | -15.8% | - |
| **fast_halfnbr** (N011's K->32) | 100 | 16,8,8 / 16,8,8 | **2.714 ms** | **-24.0%** | **-9.7%** |
| **fast_randhalf** (random only) | 100 | 16,16,16 / 16,16,8 | **2.822 ms** | **-21.0%** | **-6.2%** |
| exact | 400 | 16,16,32 / 16,16,16 | 7.411 ms | - | - |
| fast (TF32 only) | 400 | 16,16,32 / 16,16,16 | 6.697 ms | -9.6% | - |
| **fast_halfnbr** | 400 | 16,8,8 / 16,8,8 | **4.782 ms** | **-35.5%** | **-28.6%** |
| **fast_randhalf** | 400 | 16,16,16 / 16,16,8 | **5.764 ms** | **-22.2%** | **-13.9%** |

The interesting thing the L=400 row adds that N011 never measured (their
K-reduction numbers are L=100 only): **the payoff GROWS with L.** At L=100,
`halfnbr` buys 9.7% on top of TF32; at L=400 it buys 28.6%. This is consistent
with the "launch-bound, not compute-bound" diagnosis in N011/N012's kernel
census - TF32 helps a roughly-fixed set of wide GEMMs regardless of L, while
neighbour selection and the gather/argsort machinery scale with the (N, N)
distance matrix, so shrinking K removes proportionally more work as N grows.
This is exactly the kind of lever the brief asked to look for ("approximating
or skipping work that is numerically near-redundant") and it is real - the
quality cost in §3 is also real, and does not shrink with L (not measured at
L=400 here; see §5 for why).

## 3. Quality: n=160 per arm, same methodology as N015 (`stats.py`, `ss.py`)

Both levers stacked on `fast`'s TF32, **full 500/400 schedule** (`--num_steps
500 --out_steps 400`), so this isolates the neighbourhood lever exactly -
nothing here is confounded with the already-validated schedule cut. Same two
seeds N015 used (42, n=40 and 1234, n=120), pooled to n=160, so this is a
DIRECT, methodologically identical comparison against N015's already-banked
`a_exact` (n=160) baseline - no need to regenerate `exact`, since the "exact"
preset's math is unchanged since N015 (confirmed in §6).

Converged only (`exact`'s own 2/160 diverged designs dropped, matching N015's
convention):

| arm | n | CA-CA (A) | Rg (A) | helix | strand | coil | clash/design | diverged |
|---|---|---|---|---|---|---|---|---|
| a_exact (N015) | 158 | 3.7628 +- 0.0137 | 12.89 +- 1.76 | 0.566 +- 0.349 | 0.353 +- 0.315 | 0.081 +- 0.053 | 0.25 +- 0.74 | 2/160 |
| **halfnbr** | 160 | 3.7599 +- 0.0168 | 13.24 +- 2.35 | 0.613 +- 0.403 | 0.312 +- 0.367 | 0.075 +- 0.052 | **2.33 +- 3.38** | 0/160 |
| **randhalf** | 160 | 3.7642 +- 0.0177 | **14.41 +- 6.49** | 0.684 +- 0.334 | 0.248 +- 0.301 | 0.068 +- 0.058 | 0.29 +- 0.60 | 0/160 |

Welch t-test vs `a_exact` (Bonferroni over 12 tests in this family: alpha =
0.0042):

| arm | CA-CA | Rg | helix | strand | coil | clash/design |
|---|---|---|---|---|---|---|
| halfnbr | d=-0.0029 p=0.091 | d=+0.34 p=0.139 | d=+0.047 p=0.269 | d=-0.041 p=0.282 | d=-0.006 p=0.353 | **d=+2.07 t=+7.56 p=0.000\*\*** |
| randhalf | d=+0.0014 p=0.445 | d=+1.52 p=0.005 (misses Bonferroni by a hair) | **d=+0.117 t=+3.06 p=0.002\*\*** | **d=-0.105 t=-3.05 p=0.003\*\*** | d=-0.012 p=0.051 | d=+0.04 p=0.592 |

Brown-Forsythe (spread) vs `a_exact`, same Bonferroni:

| arm | CA-CA | Rg | clash |
|---|---|---|---|
| halfnbr | W=3.62 p=0.058 | **W=12.01 p=0.0006\*\*** | **W=45.79 p<0.0001\*\*** |
| randhalf | W=5.64 p=0.018 (misses) | **W=10.30 p=0.0015\*\*** | W=0.29 p=0.591 |

Two-sample KS vs `a_exact` (distribution shape; alpha=0.005 for the 10-test
family):

| arm | Rg | helix | strand |
|---|---|---|---|
| halfnbr | **D=0.188 p=0.0063** (narrowly misses) | **D=0.195 p=0.0038\*\*** | **D=0.195 p=0.0038\*\*** |
| randhalf | **D=0.212 p=0.0013\*\*** | **D=0.196 p=0.0036\*\*** | D=0.189 p=0.0056 (narrowly misses) |

Tail count (designs outside `exact`'s own mean +- 3sd Rg band, `[7.62,
18.16]` A): `a_exact` 2/160, `halfnbr` 10/160, `randhalf` **17/160**. Divergence
(CA-CA mean > 4.5 A): 0/160 for both levers vs 2/160 for `exact` (Fisher
p=0.498, not distinguishable - consistent with N015's finding that divergence
is a property of the model, not of any of these presets).

**Reading this honestly**: `halfnbr`'s failure mode is unambiguous (clashes,
huge effect, huge significance - this is the same finding N011 had at n=8,
now overwhelming). `randhalf`'s failure mode is real but different and
smaller - it does not clash, but it produces more outlier-large structures
(Rg tail more than doubles in count vs `exact`, and the effect survives
Bonferroni on both the spread test and the KS test) and shifts the
helix/strand balance by an amount that also survives Bonferroni. Sparing the
spatial neighbours avoided the acute failure (steric clashes - the network
losing track of *nearby* geometry) but not a chronic one (the network losing
some of the *long-range* signal that keeps a fold compact). Full per-design
numbers: `../N015-fast-validation/per_design.json` now also carries `halfnbr`
and `randhalf` (this run appended to the same file N015 wrote).

## 4. Divergence from JAX: the aatype count

`tests/test_sampling_parity.py` doesn't take a preset argument (it's pinned to
the eager `forward()` path, used by `exact`), so it can't directly score a
`FastDenoiser` preset; it was run as a worktree sanity check instead (§6).

For the discrete signal the brief asked for, `aatype_diff.py` compares the
sequence baked into each generated PDB, lever vs `a_exact`, matched by (seed,
design index) - same starting `torch.rand(n_draws, L, L)` draw regardless of
K, since `n_draws` depends only on model config, never on the neighbourhood
size (verified: `_blank_inputs`'s `uniform` tensor is drawn once per design at
a shape that does not mention K anywhere in the codebase).

| arm | mean residues differing (of 100) | as a fraction |
|---|---|---|
| halfnbr | 88.3 +- 5.4 | **88.3%** |
| randhalf | 84.4 +- 8.1 | **84.4%** |

Chance level for a fully independent redraw of a 20-way categorical residue is
~95%. **These numbers sit close to that ceiling, not near zero** - i.e. the
pairing-by-seed hypothesis, while mechanistically correct for the FIRST
forward call, does not survive the sampling loop: the model is chaotic
end-to-end (documented in `test_sampling_parity.py`'s own docstring - a 1e-6
perturbation to the input noise diverges ~0.58 A at the worst trajectory
point), and N011 already measured that a single forward pass under K-reduction
moves `pos` by 6.7-11.7 A (vs TF32's 0.34 A and the CUDA noise floor's
2.2e-04 A) - three to four orders of magnitude bigger than the perturbations
that stay small end to end. That single-step deviation is large enough to
fully decorrelate the trajectory by step 500.

**This is itself the sharp discrete signal the brief asked for, just not the
one I expected going in**: K-reduction does not produce "the same design with
a few flipped residues" the way the `deterministic` preset's reassociation
does (7/64 residues, per its own preset docstring) or the way TF32 is
documented to (small compared to K-reduction, per N011's dev_max table in
§2 of their RESULTS.md). It produces what is, for all practical purposes, **a
different design from the same nominal seed** - on top of the structural cost
already shown in §3. Read the two together: it's not that the network makes
90-degree-different sequence choices on an otherwise-similar backbone: the
backbones themselves measurably differ (§3), so the near-total aatype
decorrelation is the expected downstream consequence, not a separate defect.

## 5. What this does and doesn't establish, and what's next

* **L=400 quality was not run** (only speed). The brief's own established
  methodology (N015) is L=100-only, and a 160-design quality arm at L=400
  would cost more than this task's modest GPU budget really needs to spend to
  answer "is either lever recommendable" - which is already a clear no at
  L=100. If a length-adaptive version of this lever (e.g. K scales with L
  instead of a fixed absolute cut) is worth pursuing later - motivated
  directly by §2's finding that the payoff grows with L - it would need its
  own quality gate at whatever L it targets.
* **Only two points on the K/quality tradeoff curve were measured** (full K,
  "half" K~32, "randhalf" K~48/40 with the cut concentrated on the random
  component). A finer sweep (e.g. random-only at K=56/40/24, or spatial-only
  reduction as the complementary experiment to randhalf) was not run - the
  brief asked for measured speed AND quality at adequate n per candidate, and
  budget was spent validating the two most informative points (N011's
  original cut, and the most promising alternative decomposition of it)
  properly rather than spreading n thin over more points.
* **Designability was not checked** (no ProteinMPNN + folding model), same
  caveat N015 already recorded for its own arms - a structural gate rules out
  "obviously broken," not "designable."
* The two rejected levers are still in `nbr_mode.patch` for anyone who wants
  to sweep further; they should NOT be adopted as `fast_halfnbr` /
  `fast_randhalf` preset names in the shared tree as-is.

## 6. Worktree sanity: my edit is a byte-identical no-op on `exact`

`opt.NBR_MODE` defaults to `"full"`, which reproduces the original hardcoded
`(16,16,32)`/`(16,16,16)` literals exactly - a pure parameterisation, not a
behaviour change - for every preset that doesn't set `nbr_mode`. Checked, not
assumed: `run_gates.sh` (N015's own gate script) against the worktree, before
any of the above ran:

| gate | floor | measured |
|---|---|---|
| `test_model_parity` default_vp / _vp_scaled / _ve_scaled | 0.00002 A | max 0.00002 A, aatype 0/48 (all three) |
| `test_model_parity` multimotif_vp | 0.00020 A | max 0.00020 A, aatype 0/80 |
| `test_geometry_parity` | 38/38 | 38/38 |
| `test_stack_parity` | - | rel 2.206e-06, max 0.0003 A |
| `test_encoder_parity` | - | rel 1.374e-06, max 0.00001 A |
| `test_sampling_parity` | ratio gate | **8.3x**, identical to N015's own number |

`SUITE_EXIT=0`, and every figure matches N015's baseline gate run to the
printed digit - confirming the refactor that made `nbr_mode` pluggable didn't
perturb anything before the new presets turn it on.

## 7. Reproducing

```bash
git worktree add --detach /hnvme/workspace/v181bb10-salad/n018-worktree a8533a7
cd /hnvme/workspace/v181bb10-salad/n018-worktree
patch -p1 < .../N018-fast-levers/nbr_mode.patch
ln -s /hnvme/workspace/v181bb10-salad/params params
ln -s /hnvme/workspace/v181bb10-salad/ae_params ae_params
ln -s /home/hpc/v181bb/v181bb10/SALAD_torch/fixtures fixtures

sbatch run_speed_L100.sbatch      # exact/fast/fast_halfnbr/fast_randhalf, L=100
sbatch run_speed_L400.sbatch      # same, L=400
sbatch run_quality_halfnbr.sbatch # n=160 (seeds 42+1234), fast_halfnbr
sbatch run_quality_randhalf.sbatch
sbatch run_gates.sbatch           # sanity: exact preset unaffected

python aatype_diff.py <lever_dirs> <n015_designs_a_exact_dirs>
python ../N015-fast-validation/stats.py a_exact=<...> halfnbr=<...> randhalf=<...>
```

GPU cost actually used: 903984 (16m27s) + 903985 (17m33s) + 903986 (9m52s) +
903987 (10m16s) + 903988 (34s, fixtures missing, fixed) + 904027 (51s) =
**~56 minutes ~ 0.93 GPU-hours** for the whole task.
