# N015 — does the `fast` preset's halved schedule cost anything?

Measured 2026-09-26 on an H100, `default_vp-200k`, L=100. Jobs **903264**
(40/arm, seed 42, 14 min) and **903296** (120/arm, seed 1234, 20 min), pooled to
**n=160 per arm**. Both ran from a clean detached worktree at `c5ab7a0`
(`/hnvme/workspace/v181bb10-salad/n015-worktree`), not the shared checkout,
because another agent was editing `modules/{blocks,diffusion,fast}.py`,
`opt.py` and `geometry/segment.py` throughout.

## 1. Verdict

**`steps_scale=0.5` stays the `fast` default.** At n=160 per arm the halved
schedule is indistinguishable from `exact` on every metric, in mean, in spread,
and in distribution shape — and the specific failure it was suspected of (less
time to converge => less secondary structure, more extended designs) is refuted
*with power*, not merely unobserved.

No preset split, no more conservative value. `exact` is unchanged (the N015 diff
is comment-only; all gates re-run, §7).

## 2. The four arms

The preset is two knobs and they are not equally well evidenced, so they were
factorised. Same `--seed` per job; **unpaired**, because a 250-step run consumes
a different number of draws per design than a 500-step one, so seed *i* is not
the same starting noise across arms.

| arm | preset | schedule | TF32 | what it isolates |
|---|---|---|---|---|
| A `a_exact` | exact | 500/400 | no | baseline |
| B `b_tf32` | fast + explicit `--num_steps 500 --out_steps 400` | 500/400 | yes | arithmetic only |
| C `c_sched` | exact + explicit `--num_steps 250 --out_steps 200` | 250/200 | no | **trajectory only** |
| D `d_fast` | fast | 250/200 | yes | the shipped preset |

## 3. Results, n=160 per arm

Metrics are N011's `ss.py` unchanged (CA-only P-SEA secondary structure), so
they compare directly with its n=8 table. **Converged designs only** — the
diverged ones are counted in their own column, where they belong; leaving them
in puts `exact`'s Rg sd at 39.55 Å, against which no arm can differ in any
direction.

| arm | n | CA-CA (Å) | Rg (Å) | helix | strand | coil | clash/design | **diverged** |
|---|---|---|---|---|---|---|---|---|
| A exact | 158 | 3.7628 ± 0.0137 | 12.89 ± 1.76 | 0.566 ± 0.349 | 0.353 ± 0.315 | 0.081 ± 0.053 | 0.25 ± 0.74 | **2/160** |
| B TF32 only | 160 | 3.7615 ± 0.0123 | 12.89 ± 1.46 | 0.572 ± 0.322 | 0.351 ± 0.289 | 0.077 ± 0.052 | 0.18 ± 0.46 | 0/160 |
| C schedule only | 160 | 3.7627 ± 0.0141 | 12.89 ± 1.56 | 0.575 ± 0.305 | 0.345 ± 0.280 | 0.080 ± 0.048 | 0.22 ± 0.51 | 0/160 |
| D **fast** | 160 | 3.7613 ± 0.0131 | 12.90 ± 1.89 | 0.580 ± 0.308 | 0.335 ± 0.281 | 0.086 ± 0.052 | 0.17 ± 0.41 | 0/160 |

Welch t-test against `exact` (difference, t, p):

| arm | CA-CA (Å) | Rg (Å) | helix | strand | coil | clash |
|---|---|---|---|---|---|---|
| B TF32 only | −0.0013, −0.89, p=0.375 | −0.00, −0.01, p=0.995 | +0.006, +0.15, p=0.878 | −0.002, −0.06, p=0.955 | −0.004, −0.66, p=0.512 | −0.07, −1.04, p=0.300 |
| C schedule only | −0.0001, −0.08, p=0.933 | −0.00, −0.02, p=0.980 | +0.009, +0.24, p=0.810 | −0.009, −0.26, p=0.799 | −0.000, −0.06, p=0.955 | −0.03, −0.48, p=0.630 |
| D **fast** | −0.0015, −0.99, p=0.322 | +0.00, +0.02, p=0.987 | +0.013, +0.36, p=0.718 | −0.019, −0.56, p=0.579 | +0.005, +0.89, p=0.376 | −0.08, −1.16, p=0.246 |

Nothing is close. The three Rg means agree to 0.01 Å.

(The same tables for the 40-per-arm first job alone are in
`logs/stats_903264.log`; the pooled run is `logs/stats_pooled_n160.log`.)

## 4. The three things that make this an answer and not just a null

### 4.1 The null has power — stated, not assumed

A null result is worth exactly the effect it could have detected, and per-design
secondary-structure variance here is enormous (helix sd 0.35, because a design
is mostly-helix *or* mostly-sheet). Two-sided α=0.05, power 0.80, n=158/160:

| metric | sd (exact) | SE(diff) | **detectable** | largest observed abs(d) |
|---|---|---|---|---|
| CA-CA (Å) | 0.0137 | 0.0015 | **0.0041** | 0.0015 |
| Rg (Å) | 1.76 | 0.18 | **0.51** | 0.00 |
| helix | 0.349 | 0.038 | **0.106** | 0.013 |
| strand | 0.315 | 0.034 | **0.095** | 0.019 |
| coil | 0.053 | 0.006 | **0.017** | 0.005 |
| clash/design | 0.74 | 0.07 | **0.19** | 0.08 |

So the claim is: *the halved schedule does not shift helix fraction by 0.11,
strand by 0.10, Rg by 0.51 Å or CA-CA by 0.004 Å.* It is **not** a claim that
the shift is exactly zero — resolving the 0.013 helix difference that was
observed would need n≈2700. For a preset decision that is the right question: a
0.013 helix shift is not a reason to give up 2× throughput, a 0.11 one would be.

The sign is wrong for the hypothesis anyway: the halved-schedule arms have
**more** helix (+0.009 and +0.013) and **identical** Rg — not less and larger.

### 4.2 The mean is not the only way a trajectory change can hurt

Halving the schedule could leave every mean in place and still widen the
distribution; a design pipeline is judged on its tail. Welch cannot see that, so
two more tests:

* **Brown-Forsythe** (median-centred Levene) on the spread: Rg p=0.99 / 0.83 /
  0.63, CA-CA p=0.088 / 0.71 / 0.20. Helix and strand come up *narrower* in the
  two halved arms (helix sd 0.349 -> 0.305, p=0.0064; strand p=0.014) — the
  wrong direction to be a degradation, and neither survives Bonferroni over the
  18 tests in that family (α=0.0028).
* **Two-sample Kolmogorov–Smirnov**, the distribution-free omnibus: every arm,
  every metric, **p >= 0.13** (smallest: `fast` CA-CA, D=0.129, p=0.126). The
  halved schedule does not sample a differently-shaped population of designs.

### 4.3 n=40 would have got this wrong, in both directions

This is why the job was re-run at 120/arm instead of stopping at the 40 asked
for.

| | n=8 (N011 §4) | n=40 (job 903264) | n=160 (pooled) |
|---|---|---|---|
| helix, exact -> fast | 0.605 -> 0.591 | 0.560 -> 0.584 | 0.566 -> 0.580 |
| Rg, exact -> fast | 12.50 -> 12.92 | 12.77 -> 13.05 | 12.89 -> 12.90 |
| Rg sd, exact -> fast | 0.78 -> 0.86 | **0.93 -> 2.17** | 1.76 -> 1.89 |
| worst single Rg, fast | — | 25.1 Å | 25.1 Å (exact's worst: **28.9 Å**) |
| diverged, exact / fast | 0 / 0 | 0 / 0 | **2 / 0** |

At n=8 the means moved in the feared direction; that was noise. At n=40 the
means were flat but `fast`'s Rg **spread** was 2.3× `exact`'s, off three
designs — which looks exactly like a real tail effect and is also noise: the
seed-42 exact batch simply happened to be tight (sd 0.93 against its own pooled
1.76). Both would have been reportable findings. Neither survived.

The standing rule from N001 §10 — *with per-design SSE variance this large, n=5
says nothing* — needs raising: **n=40 says nothing about the tail either.**

## 5. Diverged designs: a model failure mode, and it is `exact`'s too

A design is DIVERGED when its mean CA-CA bond exceeds 4.5 Å. Upstream JAX
produced 1 in 40 (N001 §10), which is why the column exists.

| arm | diverged | worst | Fisher exact vs exact |
|---|---|---|---|
| A exact | **2/160** | `design_39` CA-CA 28.3 Å, Rg 408 Å, 61 clashes; `design_60` CA-CA 13.7 Å, Rg 321 Å | — |
| B TF32 only | 0/160 | — | p=0.498 |
| C schedule only | 0/160 | — | p=0.498 |
| D fast | 0/160 | — | p=0.498 |

Both of `exact`'s are in the seed-1234 batch
(`/hnvme/workspace/v181bb10-salad/n015_designs2/a_exact/design_{39,60}.pdb`).
At 2 vs 0 in 160 this is not a quality claim in either direction — but it does
settle the direction of the worry: **the deviating arms did not add a single
divergence, and the only arm that diverged is the exact one.**

One case the CA-CA filter misses: `a_exact/design_72` has a perfect 3.739 Å bond
length and Rg 28.9 Å — a correctly-bonded, fully extended chain. It is the most
extended design in the whole study, and it is in the `exact` arm. A downstream
pipeline should filter on Rg as well as CA-CA, on either preset.

## 6. What was NOT tested, and why it bounds the verdict

* **Designability.** Every metric here is structural (bond geometry, Rg, CA-only
  P-SEA secondary structure). None of it is ProteinMPNN + a folding model. A
  structural gate cannot tell you a backbone is designable; only that it is not
  obviously broken. 250/200 is now well evidenced *on that gate*, at n=160 —
  much stronger than N011's n=8, and still not a designability result.
* **125/100 ("faster").** N011 §4 found it passes the same structural gate at
  n=8. It is deliberately **not** offered as a preset: a 4× cut is where a
  designability check stops being optional. The command line already reaches it
  — `--fast --preset fast --num_steps 125 --out_steps 100` — without a preset
  blessing it.
* **L=100 only, `default_vp` only.** The schedule interacts with length in
  principle. N011's length sweep covers speed at L=50…400; quality does not.

## 7. Hard gates — `exact` unchanged

The N015 code change is **comment-only** inside `PRESETS["fast"]`: no preset
value moved and `exact` is untouched. Gates ran on a clean worktree holding
exactly `c5ab7a0` + that diff (`run_gates.sh`, `logs/gates_n015.log`), so they
measure this change and not the other agent's concurrent edits to the shared
checkout.

| gate | floor | measured |
|---|---|---|
| `test_model_parity` `default_vp` | 0.00002 Å | mean 0.00000, **max 0.00002 Å**, aatype 0/48 |
| `test_model_parity` `default_vp_scaled` | 0.00002 Å | mean 0.00000, **max 0.00002 Å**, aatype 0/48 |
| `test_model_parity` `default_ve_scaled` | 0.00002 Å | mean 0.00000, **max 0.00002 Å**, aatype 0/48 |
| `test_model_parity` `multimotif_vp` | 0.00020 Å | mean 0.00001, **max 0.00020 Å**, aatype 0/80 |
| `test_geometry_parity` | 38/38 | **38/38 passed** |
| `test_stack_parity` | — | rel 2.206e-06, max 0.0003 Å |
| `test_encoder_parity` | — | local rel 1.374e-06, coords max 0.00001 Å |
| `test_sampling_parity` | ratio gate | final/noise-floor **8.3×** (mean 0.05374 Å vs floor 0.00650 Å) |

`SUITE_EXIT=0`, and `diff` against the same suite on unmodified `c5ab7a0`
(`logs/gates_baseline_HEAD.log`) is empty — every figure matches to the printed
digit.

## 8. Reproducing

```bash
sbatch .agent-research/results/N015-fast-validation/run_arms.sbatch   # 40/arm,  seed 42
sbatch .agent-research/results/N015-fast-validation/run_more.sbatch   # 120/arm, seed 1234
D=/hnvme/workspace/v181bb10-salad/n015_designs
D2=/hnvme/workspace/v181bb10-salad/n015_designs2
python .agent-research/results/N015-fast-validation/stats.py \
  a_exact=$D/a_exact,$D2/a_exact   b_tf32=$D/b_tf32,$D2/b_tf32 \
  c_sched=$D/c_sched,$D2/c_sched   d_fast=$D/d_fast,$D2/d_fast
bash .agent-research/results/N015-fast-validation/run_gates.sh <checkout>
```

GPU cost: 14:31 + 19:54 = **0.57 GPU-hours** for 640 designs, each job one H100,
both under the one-hour-per-job cap.
