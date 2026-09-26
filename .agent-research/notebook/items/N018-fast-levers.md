# N018 · fast-levers

type: N018 · date: 2026-09-26 · status: done

## Goal

Find numerical-relaxation levers that shrink `fast` preset's per-STEP kernel
time (not the step count - that was already spent by N011/N015's
`steps_scale=0.5`), measure both speed and structural-quality cost at
adequate n, and assess whether N011's negative results (K64->32, bf16 GEMM,
combo kernels) were measured badly or genuinely don't pay.

## Plan

1. Read N011/N012/N013/N014/N015 first; judge N011's methodology per-item.
2. Own snapshot: detached worktree at HEAD (`a8533a7`), since
   blocks/diffusion/fast/opt/segment.py are being actively edited in the
   shared tree.
3. Candidate levers: reproduce N011's K64->32 cut exactly (re-test its
   quality at n=160, not n=8); propose+test a milder cut (halve only the
   RANDOM neighbour component, leave spatial untouched).
4. Speed: production `FastDenoiser(preset=...)`, L=100 AND L=400,
   `steps_scale=1.0` fixed, 3 reps, one shape (K) per process.
5. Quality: N015's own methodology (`stats.py`/`ss.py`), n=160 (pooled
   seed 42 n=40 + seed 1234 n=120, SAME seeds as N015's `a_exact` so no
   baseline regeneration needed - preset math unchanged since N015).
6. Aatype divergence: pair by (seed, index) against N015's `a_exact` PDBs.
7. Sanity: N015's `run_gates.sh` against the worktree (confirm `nbr_mode`
   defaulting to "full" is a byte-identical no-op).

## Changes

New env-gated experiment-only knob `opt.NBR_MODE` (`full`/`half`/`randhalf`),
wired into `Diffusion.forward_core`'s `neighbour_fn`, plus two new
`FastDenoiser` presets (`fast_halfnbr`, `fast_randhalf`). **This worktree
only** - never landed in the shared tree, patch preserved at
`results/N018-fast-levers/nbr_mode.patch` (verified `patch -p1 --dry-run`
clean against `a8533a7`). Worktree deleted after use (inode grace period).

## Evidence

5 SLURM jobs, H100, ~0.93 GPU-h total (903984/85 speed L100/L400, 903986/87
quality halfnbr/randhalf n=160, 903988+904027 gates - first attempt failed on
a missing `fixtures/` symlink, fixed and re-run cheaply). Full numbers, stats
tables (Welch/Brown-Forsythe/KS/Bonferroni) and the aatype-divergence output
in `results/N018-fast-levers/RESULTS.md`.

## Outcome

**Neither lever recommended.** Both are genuine per-step speedups
(halfnbr -9.7%/-28.6% beyond TF32 at L=100/L=400; randhalf -6.2%/-13.9%) and
both cost real structure at n=160: halfnbr reproduces N011's clash blowup far
more decisively (2.33 vs 0.25 clash/design, p=0.000, spread p<0.0001);
randhalf avoids the clash failure but shows a Bonferroni-surviving Rg-tail
widening (17/160 vs 2/160 outliers, spread p=0.0015, KS p=0.0013) and a
helix/strand composition shift (p=0.002/0.003). N011's own negative results
were assessed as correctly measured (the bf16/combo_kernels reasoning holds,
though combo_kernels' actual number was left untranscribed in their
RESULTS.md - surfaced here: -1.0%, not worth it either way); the one gap
(K-reduction quality at n=8) was correctly self-flagged as insufficient by
N011 and, redone at n=160, makes the rejection MORE decisive, not less. New
finding not in any prior task: the K-reduction payoff grows sharply with L
(9.7% -> 28.6% from L=100 to L=400), which the shipped preset's L=100-only
quality gate never had reason to surface.

## Outputs

- `results/N018-fast-levers/RESULTS.md` - full writeup
- `nbr_mode.patch`, `opt_modified.py`, `diffusion_modified.py`,
  `fast_modified.py` - the tested code, since the worktree was deleted
- `bench_speed.py`, `run_speed_L100.sbatch`, `run_speed_L400.sbatch`
- `run_quality_halfnbr.sbatch`, `run_quality_randhalf.sbatch`, `run_gates.sbatch`
- `aatype_diff.py`
- `logs/` - all 6 job logs + `stats_n018.log` + `aatype_diff_output.log`
- `../N015-fast-validation/per_design.json` - now also carries `halfnbr` and
  `randhalf` rows (stats.py append; N015's own `a_exact`/`b_tf32`/etc. rows
  untouched)
- External (deleted after use): `/hnvme/workspace/v181bb10-salad/n018-worktree`,
  `/hnvme/workspace/v181bb10-salad/n018_designs` (320 PDBs, 21M)

## Next

If a length-adaptive neighbourhood (K scaling with L instead of a fixed cut)
is ever proposed - motivated by the L=400 speed finding - it needs its own
quality gate at whatever L it targets; nothing here licenses extrapolating
the L=100 quality verdict to L=400.
