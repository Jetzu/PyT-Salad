# N015 · Fast preset: 40-design validation of the halved schedule

type: exp · date: 2026-09-26 · status: done

## Goal

N011 shipped `fast` = TF32 + `steps_scale=0.5`. The two knobs are not equally
well evidenced: TF32 is arithmetic (3.4e-01 A, ~1500x the 2.2e-04 A CUDA noise
floor, N010) and does not alter the trajectory's structure, whereas **halving
the schedule changes the trajectory itself** and is worth ~50% of the run. Its
quality evidence is n=8 (N011 §4). Earlier in this project an n=5 structural
comparison produced an apparent helix skew that a 40-design run (N001) showed
to be pure sampling noise, so n=8 is not enough to accept a trajectory change.

Decide, at n=40 per arm, whether `steps_scale=0.5` stays the `fast` default.

## Plan
<!-- exp: hypothesis + expected result · ONE variable · baseline -->

**Hypothesis.** A halved schedule gives the trajectory half as many steps to
converge, so if it costs anything it should show up as LESS secondary structure
(lower helix + strand), a LARGER radius of gyration (more extended), or more
DIVERGED designs. N011's n=8 hinted at exactly this shape - helix 0.605 ->
0.591, Rg 12.50 -> 12.92 - but at n=8 neither is separable from noise.

**Design.** 40 de-novo designs at L=100, `default_vp-200k`, same `--seed 42`,
four arms that factorise the preset:

| arm | preset | schedule | TF32 |
|---|---|---|---|
| A `a_exact` | exact | 500/400 | no |
| B `b_tf32` | fast + explicit `--num_steps 500 --out_steps 400` | 500/400 | yes |
| C `c_sched` | exact + explicit `--num_steps 250 --out_steps 200` | 250/200 | no |
| D `d_fast` | fast | 250/200 | yes |

B and C isolate one knob each; D is the shipped preset; A is the baseline.

**Metrics** (N011's `ss.py`, unchanged, so the numbers compare directly to its
n=8 table): CA-CA bond mean, Rg, P-SEA helix / strand / coil fraction, CA-CA
clashes, and the count of DIVERGED designs (CA-CA mean > 4.5 A - upstream JAX
itself produced 1 diverged design in 40, N001, so this column is not decorative).

**Test.** Unpaired Welch t-test of each arm against A. Unpaired on purpose: a
250-step run consumes a different number of draws from the seeded generator per
design than a 500-step run, so the same `--seed` does not give design i the same
starting noise across arms. A paired test would be wrong here.

**Isolation.** Another agent is editing `modules/{blocks,diffusion}.py`,
`opt.py` and `geometry/segment.py` in the main checkout (N013/N014, batched
condition-branch GEMMs + a deterministic segment sum). The job therefore runs
from a clean detached worktree at `c5ab7a0`
(`/hnvme/workspace/v181bb10-salad/n015-worktree`), so a 20-minute measurement
cannot read a half-written file.

## Changes

`salad_torch/modules/fast.py`, `PRESETS["fast"]` - COMMENT ONLY. The n=8
quality paragraph is replaced by the n=160 table, the power statement and the
diverged-design finding, and the `steps_scale=0.5` line now carries the
decision (stays 0.5; no split, no 0.75; no "faster" preset at 125/100 without a
designability check). No preset value moved. `exact` untouched.

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

* job **903264**, h100, 1 GPU, 14:31, `run_arms.sbatch` - 40/arm, seed 42.
  PDBs -> `/hnvme/workspace/v181bb10-salad/n015_designs/{a_exact,b_tf32,c_sched,d_fast}/`
* job **903296**, h100, 1 GPU, 19:54, `run_more.sbatch` - 120/arm, seed 1234,
  submitted because the n=40 means were flat but `fast`'s Rg sd read 2.3x
  `exact`'s off 3 designs, which n=40 cannot separate from noise.
  PDBs -> `/hnvme/workspace/v181bb10-salad/n015_designs2/...`
* 0.57 GPU-h total for 640 designs; both jobs under the 1 h/job cap.

Pooled n=160/arm, converged designs, `exact` = A:

| arm | CA-CA (A) | Rg (A) | helix | strand | diverged |
|---|---|---|---|---|---|
| A exact | 3.7628 +- 0.0137 | 12.89 +- 1.76 | 0.566 | 0.353 | **2/160** |
| B TF32 only | 3.7615 +- 0.0123 | 12.89 +- 1.46 | 0.572 | 0.351 | 0/160 |
| C schedule only | 3.7627 +- 0.0141 | 12.89 +- 1.56 | 0.575 | 0.345 | 0/160 |
| D fast | 3.7613 +- 0.0131 | 12.90 +- 1.89 | 0.580 | 0.335 | 0/160 |

Welch vs A: all p >= 0.24. Two-sample KS: all p >= 0.13. Brown-Forsythe on the
spread: nothing survives Bonferroni (alpha=0.0028 over 18). Fisher on diverged:
p=0.498. Power at n=160: detectable 0.106 helix / 0.095 strand / 0.51 A Rg /
0.0041 A CA-CA, against largest observed 0.013 / 0.019 / 0.00 / 0.0015.

Hard gates on a clean worktree at `c5ab7a0` + this diff
(`logs/gates_n015.log`), byte-identical to the same suite on unmodified
`c5ab7a0` (`logs/gates_baseline_HEAD.log`): default_vp / default_vp_scaled /
default_ve_scaled max 0.00002 A, multimotif_vp max 0.00020 A, aatype 0/48 and
0/80; geometry 38/38; stack rel 2.206e-06 max 0.0003 A; encoder coords max
0.00001 A; sampling final/noise-floor 8.3x. `SUITE_EXIT=0`.

## Outcome

**`steps_scale=0.5` STAYS the `fast` default.** The halved schedule - the only
knob in the preset that changes the trajectory rather than the arithmetic - is
indistinguishable from `exact` at n=160 in mean, spread and distribution shape.
The feared direction (less secondary structure, more extended) is refuted with
power, and the sign is opposite: helix goes UP by 0.013.

Two things worth more than the verdict:

1. **n=40 is not enough either.** N011's n=8 helix/Rg shift was noise; so was an
   n=40 rerun showing `fast`'s Rg spread at 2.3x `exact`'s off 3 designs. Both
   would have been reportable findings. This raises N001 section 10's "n=5 says
   nothing" to n~100 for anything about the tail.
2. **Divergence is the MODEL's failure mode, not the preset's.** 2/160 on
   `exact` (worst CA-CA 28.3 A, Rg 408 A), 0/160 on all three deviating arms.
   And `a_exact/design_72` - CA-CA 3.739 A, Rg 28.9 A - is a correctly-bonded
   fully extended chain that a CA-CA filter does not catch, so downstream
   filtering needs Rg too.

NOT established: designability. Every metric here is structural; no
ProteinMPNN + folding model was run, which is why 125/100 is still not offered
as a preset.

## Outputs
<!-- files in .agent-research/results/N015-fast-validation/ + external absolute paths -->

- `.agent-research/results/N015-fast-validation/run_arms.sbatch` - the four arms
- `.agent-research/results/N015-fast-validation/stats.py` - summary + Welch tests
- `.agent-research/results/N015-fast-validation/per_design.json` - per-design metrics
- `.agent-research/results/N015-fast-validation/RESULTS.md` - the write-up
- `.agent-research/results/N015-fast-validation/run_more.sbatch` - the n=120 rerun
- `.agent-research/results/N015-fast-validation/run_gates.sh` - the hard gates
- `logs/{arms_903264,more_903296,stats_903264,stats_pooled_n160,gates_baseline_HEAD,gates_n015}.log`
- designs (40 MB, 640 PDBs, kept out of the repo):
  `/hnvme/workspace/v181bb10-salad/n015_designs{,2}/`

## Next

- Designability (ProteinMPNN + a folding model) is the gate none of this port's
  quality work has run. It is what 125/100 needs, and what would turn "the
  halved schedule is structurally indistinguishable" into "the halved schedule
  is free".
- `generate.py` prints CA-CA per design but does not flag a diverged one. A
  one-line warning on CA-CA > 4.5 A or Rg > 3 sd would have made the 2/160 here
  visible without a post-hoc script. Not done: outside this task's scope.
