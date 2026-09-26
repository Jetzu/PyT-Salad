"""N015 - the 40-per-arm statistical check on the `fast` preset's two knobs.

Reuses N011's `ss.py` verbatim for the per-design metrics (CA-only P-SEA
secondary structure, CA-CA bond length, Rg, clash count) so the numbers are
directly comparable with the n=8 table in N011's RESULTS.md.

What this adds is the part n=8 could not answer:
  * mean +- sd over 40 designs per arm,
  * the DIVERGED count (CA-CA mean > 4.5 A - upstream JAX itself produced 1 in
    40, see N001's distributional check), and
  * an unpaired Welch t-test of each arm against `exact`, with p-values.

The arms are unpaired on purpose: a 250-step run draws a different number of
randoms per design than a 500-step run, so the same `--seed` does NOT give
design i the same starting noise across arms. Welch (unequal variance) is the
right test; a paired test would be wrong.

Usage:  python stats.py <a_exact> <b_tf32> <c_sched> <d_fast>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "N011-fast-preset"))
from ss import summarise  # noqa: E402

DIVERGED_CA = 4.5           # CA-CA mean above this = the design fell apart
METRICS = [("ca_mean", "CA-CA (A)", 4), ("rg", "Rg (A)", 2),
           ("helix", "helix", 3), ("strand", "strand", 3),
           ("coil", "coil", 3), ("clash", "clash/design", 2)]


def load(d):
    rows = summarise(sorted(glob.glob(os.path.join(d, "*.pdb"))))
    for r in rows:
        r["diverged"] = bool(r["ca_mean"] > DIVERGED_CA or r["nan"])
    return rows


def col(rows, k, converged_only=False):
    sel = [r for r in rows if not (converged_only and r["diverged"])]
    return np.array([float(r[k]) for r in sel])


def welch(a, b):
    """Welch t-test of `a` against baseline `b`; returns (t, p, df)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    t, p = sps.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t), float(p), float(df)


def main(dirs):
    """Each argument is a directory, or `name=dir1,dir2` to POOL directories
    into one arm. Pooling is how the seed-42 n=40 batch and the seed-1234 n=120
    batch become one n=160 arm: the designs are i.i.d. draws from the same
    sampler, so which seed produced them is not a factor in any test here."""
    arms = {}
    for spec in dirs:
        if "=" in spec:
            nm, paths = spec.split("=", 1)
            rows = []
            for d in paths.split(","):
                rows += load(d)
            arms[nm] = rows
        else:
            arms[os.path.basename(spec.rstrip("/"))] = load(spec)
    names = list(arms)
    base = names[0]

    for converged_only in (False, True):
        tag = ("CONVERGED ONLY (diverged designs dropped)" if converged_only
               else "ALL DESIGNS")
        print(f"\n================ {tag} ================")
        print(f"{'arm':<10} {'n':>3} " + " ".join(
            f"{lab:>16}" for _, lab, _ in METRICS) + f"{'diverged':>10}")
        for nm, rows in arms.items():
            n = len(col(rows, "ca_mean", converged_only))
            cells = []
            for k, _, prec in METRICS:
                v = col(rows, k, converged_only)
                cells.append(f"{v.mean():.{prec}f}+-{v.std(ddof=1):.{prec}f}".rjust(16))
            ndiv = sum(r["diverged"] for r in rows)
            print(f"{nm:<10} {n:>3} " + " ".join(cells) + f"{ndiv:>10}")

        print(f"\n-- Welch t-test vs {base} (p < 0.05 flagged *, p < 0.01 **)")
        print(f"{'arm':<10} " + " ".join(f"{lab:>26}" for _, lab, _ in METRICS))
        for nm in names[1:]:
            cells = []
            for k, _, prec in METRICS:
                a = col(arms[nm], k, converged_only)
                b = col(arms[base], k, converged_only)
                t, p, _ = welch(a, b)
                star = "**" if p < 0.01 else ("*" if p < 0.05 else " ")
                d = a.mean() - b.mean()
                cells.append(f"d={d:+.{prec}f} t={t:+5.2f} p={p:.3f}{star}".rjust(26))
            print(f"{nm:<10} " + " ".join(cells))

    # per-design dump, so the table can be recomputed without the GPU
    out = {nm: [{k: r[k] for k in
                 ("path", "n", "nan", "ca_mean", "ca_min", "ca_max", "rg",
                  "helix", "strand", "coil", "clash", "diverged")}
                for r in rows] for nm, rows in arms.items()}
    with open(os.path.join(HERE, "per_design.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nper-design metrics -> {os.path.join(HERE, 'per_design.json')}")

    # What n=40 can and cannot see. A null result is only worth as much as the
    # effect it could have detected, and the per-design SSE variance here is
    # large (N001 §10 measured helix sd ~0.26 across 40 designs), so this is
    # not a formality.
    # Everything below is computed on the CONVERGED subset. The exact arm's
    # two diverged designs (CA-CA 28.3 A, Rg 408 A) put its all-designs sd at
    # 39.55 A, against which no other arm can differ in any direction - an
    # all-designs spread test is arithmetic about two outliers, not about the
    # knob. The diverged designs are counted in the table above, which is where
    # they belong.
    CO = True
    n0 = len(col(arms[base], "rg", CO))
    print(f"\n-- what n={n0} (converged) could have detected "
          f"(two-sided alpha=0.05, power 0.80)")
    print(f"{'metric':<14} {'sd(exact)':>10} {'SE(diff)':>10} {'MDE':>10} "
          f"{'largest |observed d|':>22}")
    for k, lab, prec in METRICS:
        b = col(arms[base], k, CO)
        obs = max(abs(col(arms[nm], k, CO).mean() - b.mean()) for nm in names[1:])
        sds = [col(arms[nm], k, CO).std(ddof=1) for nm in names]
        se = float(np.sqrt(sum(s ** 2 / len(b) for s in sds[:2])))
        mde = 2.802 * se          # (z_{0.975} + z_{0.80}) = 1.960 + 0.842
        print(f"{lab:<14} {b.std(ddof=1):>10.{prec}f} {se:>10.{prec}f} "
              f"{mde:>10.{prec}f} {obs:>22.{prec}f}")

    # A trajectory change can leave every MEAN where it was and still widen the
    # distribution - which is a quality change, because a design pipeline is
    # judged on its tail, not its centre. Welch's t-test is blind to that by
    # construction, so test the SPREAD explicitly: Brown-Forsythe (Levene
    # centred on the median, i.e. robust to the outliers that are the point).
    print("\n-- spread vs a_exact, CONVERGED only: Brown-Forsythe (median-centred Levene)")
    print(f"{'arm':<10} " + " ".join(f"{lab:>30}" for _, lab, _ in METRICS))
    for nm in names[1:]:
        cells = []
        for k, _, prec in METRICS:
            a, b = col(arms[nm], k, CO), col(arms[base], k, CO)
            W, p = sps.levene(a, b, center="median")
            star = "**" if p < 0.01 else ("*" if p < 0.05 else " ")
            cells.append(f"sd {a.std(ddof=1):.{prec}f}/{b.std(ddof=1):.{prec}f} "
                         f"W={W:5.2f} p={p:.4f}{star}".rjust(30))
        print(f"{nm:<10} " + " ".join(cells))

    # ... and count the tail directly. "3 sd of the exact arm" is an arbitrary
    # line, but it is the same line for every arm, which is what matters.
    print("\n-- tail counts, CONVERGED only (threshold = exact mean +- 3 sd of the EXACT arm)")
    for k, lab, prec in METRICS[:2]:
        b = col(arms[base], k, CO)
        lo, hi = b.mean() - 3 * b.std(ddof=1), b.mean() + 3 * b.std(ddof=1)
        row = " ".join(f"{nm} {int(((col(arms[nm], k, CO) < lo) | (col(arms[nm], k, CO) > hi)).sum())}"
                       for nm in names)
        print(f"{lab:<14} outside [{lo:.{prec}f}, {hi:.{prec}f}]: {row}")

    # Welch tests the mean, Levene the spread. Neither sees a change of SHAPE -
    # and "does halving the schedule sample a different distribution of
    # designs?" is a question about the whole distribution. Two-sample
    # Kolmogorov-Smirnov is the distribution-free omnibus for it.
    print("\n-- distribution shape vs a_exact, CONVERGED only: two-sample KS")
    print(f"{'arm':<10} " + " ".join(f"{lab:>22}" for _, lab, _ in METRICS[:5]))
    for nm in names[1:]:
        cells = []
        for k, _, prec in METRICS[:5]:
            D, p = sps.ks_2samp(col(arms[nm], k, CO), col(arms[base], k, CO))
            star = "**" if p < 0.01 else ("*" if p < 0.05 else " ")
            cells.append(f"D={D:.3f} p={p:.4f}{star}".rjust(22))
        print(f"{nm:<10} " + " ".join(cells))

    # The count columns are not t-testable at these rates; Fisher's exact test
    # is. `diverged` is the one that decides whether a knob may ship.
    print("\n-- diverged designs: Fisher exact vs a_exact")
    nb, db = len(arms[base]), sum(r["diverged"] for r in arms[base])
    for nm in names[1:]:
        na, da = len(arms[nm]), sum(r["diverged"] for r in arms[nm])
        _, p = sps.fisher_exact([[da, na - da], [db, nb - db]])
        star = "**" if p < 0.01 else ("*" if p < 0.05 else " ")
        print(f"{nm:<10} {da}/{na} vs {db}/{nb}   p={p:.4f}{star}")

    ntest = 6 * (len(names) - 1)
    print(f"\n-- NOTE: {ntest} mean tests + {ntest} spread tests + "
          f"{5 * (len(names) - 1)} KS tests were run. Bonferroni over the "
          f"{ntest} tests of one family is alpha={0.05 / ntest:.4f}; a single "
          f"p in the 0.01-0.05 range does not survive it.")

    # the extremes matter as much as the means for a trajectory change
    print("\n-- tails (worst single design per arm)")
    for nm, rows in arms.items():
        ca = col(rows, "ca_mean")
        print(f"{nm:<10} CA-CA min {min(r['ca_min'] for r in rows):.3f} "
              f"max {max(r['ca_max'] for r in rows):.3f}   "
              f"per-design CA-CA mean range {ca.min():.4f}..{ca.max():.4f}   "
              f"Rg range {col(rows,'rg').min():.1f}..{col(rows,'rg').max():.1f}  "
              f"max clash {max(r['clash'] for r in rows)}")


if __name__ == "__main__":
    main(sys.argv[1:])
