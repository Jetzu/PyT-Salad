"""N035 evidence: does --dssp "parent/child_1/child_2" actually do what it
claims - matching secondary structure per state, and matching TERTIARY
structure (real coordinates) at same_mask positions?

No GPU/torch dependency: pure numpy + the existing P-SEA CA-only assigner
(`.agent-research/results/N011-fast-preset/ss.py`, reused verbatim per the
task's instruction, not reimplemented). Run this AFTER `scripts/generate.py`
has produced:

  * a COUPLED run:   --dssp "<parent>/<child_1>/<child_2>"  -> out_dir/
      design_{n}_parent_{step}.pdb, design_{n}_child_1_{step}.pdb,
      design_{n}_child_2_{step}.pdb
  * an UNCOUPLED control: three INDEPENDENT single-state runs
      (--dssp "<parent>", --dssp "<child_1>", --dssp "<child_2>" separately)
      -> parent_dir/design_{n}.pdb, child1_dir/design_{n}.pdb,
      child2_dir/design_{n}.pdb

Usage:
  python evidence.py <parent_dssp> <child1_dssp> <child2_dssp> \\
      --coupled coupled_dir --step 400 \\
      --unc_parent unc_parent_dir --unc_child1 unc_child1_dir --unc_child2 unc_child2_dir \\
      --n 4
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

import numpy as np

_SS_PATH = os.path.join(os.path.dirname(__file__), "..", "N011-fast-preset", "ss.py")
_spec = importlib.util.spec_from_file_location("n011_ss", _SS_PATH)
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)  # provides ss.read_ca, ss.psea


def same_mask_and_split(parent: str, child_1: str) -> tuple[np.ndarray, int]:
    """`eval_split_change.py::parse_parent`'s same_mask/parent_split, without
    needing salad_torch/torch at all - see notes/specs/07_multistate_dssp.md
    §1.1. Deliberately duplicated (3 lines) rather than importing the torch
    package, so this analysis script has no GPU/torch dependency.
    """
    same_mask = np.array([c.isupper() for c in parent], dtype=bool)
    return same_mask, len(child_1)


def psea_agreement(requested: str, ca: np.ndarray) -> dict:
    """Fraction of non-'_' requested positions where P-SEA agrees.

    'H'/'E' (either case) must match P-SEA's own 'H'/'E'; 'L'/'l' must match
    P-SEA's 'c' (coil) - psea has no separate loop/unspecified distinction,
    so both "explicit loop" and "don't care" upstream inputs collapse to
    "not H, not E" on the assigned side; only the REQUESTED side keeps that
    distinction (a requested '_' is skipped entirely, never scored).
    """
    assigned = ss.psea(ca)
    n = min(len(requested), len(assigned))
    total = correct = 0
    for i in range(n):
        req = requested[i]
        if req == "_":
            continue
        total += 1
        req_u = req.upper()
        ok = (assigned[i] == req_u) if req_u in "HE" else (assigned[i] == "c")
        correct += int(ok)
    return dict(total=total, correct=correct,
               frac=(correct / total if total else float("nan")))


def kabsch_rmsd(mobile: np.ndarray, fixed: np.ndarray) -> float:
    """CA-only Kabsch-superposed RMSD, `mobile` moved onto `fixed`. Equal-
    length, 1:1 correspondence (residue i of `mobile` <-> residue i of
    `fixed`) - the same correspondence `index_align` relies on.
    """
    assert mobile.shape == fixed.shape and mobile.shape[1] == 3
    mc = mobile - mobile.mean(axis=0)
    fc = fixed - fixed.mean(axis=0)
    h = mc.T @ fc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    corr = np.diag([1.0, 1.0, d])
    rot = vt.T @ corr @ u.T
    mobile_aligned = mc @ rot.T
    return float(np.sqrt(((mobile_aligned - fc) ** 2).sum(axis=-1).mean()))


def _self_test() -> None:
    """A known rotation+translation must Kabsch-superpose to ~0 RMSD, and an
    unrelated random cloud must not - sanity-checks kabsch_rmsd's sign/
    transpose conventions before it's trusted for the real evidence numbers.
    """
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20, 3))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    moved = pts @ rot.T + np.array([5.0, -2.0, 1.0])
    r = kabsch_rmsd(moved, pts)
    assert r < 1e-8, f"self-test failed: {r}"
    other = rng.normal(size=(20, 3)) * 5
    r2 = kabsch_rmsd(other, pts)
    assert r2 > 1.0, f"self-test failed (expected large RMSD): {r2}"
    print(f"[self-test] identity-under-rigid-transform RMSD={r:.2e}, "
         f"unrelated-cloud RMSD={r2:.2f}  OK")


def find_designs(pattern: str, n: int) -> list[str]:
    paths = sorted(glob.glob(pattern))
    return paths[:n] if n else paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parent_dssp")
    ap.add_argument("child_1_dssp")
    ap.add_argument("child_2_dssp")
    ap.add_argument("--coupled", required=True)
    ap.add_argument("--unc_parent", required=True)
    ap.add_argument("--unc_child1", required=True)
    ap.add_argument("--unc_child2", required=True)
    ap.add_argument("--step", type=int, default=400)
    ap.add_argument("--n", type=int, default=0, help="0 = all found")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "evidence_results.json"))
    args = ap.parse_args()

    _self_test()

    same_mask, split = same_mask_and_split(args.parent_dssp, args.child_1_dssp)
    m1, m2 = same_mask[:split], same_mask[split:]
    print(f"parent={len(args.parent_dssp)}aa child_1={len(args.child_1_dssp)}aa "
         f"child_2={len(args.child_2_dssp)}aa  split={split}  "
         f"tertiary-coupled: {int(m1.sum())} (of child_1) + {int(m2.sum())} (of child_2)"
         f" = {int(same_mask.sum())} total")

    requested = dict(parent=args.parent_dssp, child_1=args.child_1_dssp,
                     child_2=args.child_2_dssp)

    # -------------------------------------------------------------- coupled
    parent_paths = find_designs(os.path.join(args.coupled, f"design_*_parent_{args.step}.pdb"), args.n)
    designs = sorted({os.path.basename(p).split("_")[1] for p in parent_paths}, key=int)
    print(f"coupled run: {len(designs)} design(s) found in {args.coupled}")

    ss_results = {"parent": [], "child_1": [], "child_2": []}
    rmsd_coupled = {"child_1": [], "child_2": []}

    for d in designs:
        cas = {}
        for state in ("parent", "child_1", "child_2"):
            path = os.path.join(args.coupled, f"design_{d}_{state}_{args.step}.pdb")
            ca = ss.read_ca(path)
            cas[state] = ca
            agr = psea_agreement(requested[state], ca)
            ss_results[state].append(agr["frac"])
        p, c1, c2 = cas["parent"], cas["child_1"], cas["child_2"]
        p1, p2 = p[:split], p[split:]
        if m1.sum() >= 3:
            rmsd_coupled["child_1"].append(kabsch_rmsd(c1[m1], p1[m1]))
        if m2.sum() >= 3:
            rmsd_coupled["child_2"].append(kabsch_rmsd(c2[m2], p2[m2]))

    # ------------------------------------------------------------ uncoupled
    unc_dirs = dict(parent=args.unc_parent, child_1=args.unc_child1, child_2=args.unc_child2)
    unc_cas = {"parent": [], "child_1": [], "child_2": []}
    for state, d in unc_dirs.items():
        paths = find_designs(os.path.join(d, "design_*.pdb"), args.n)
        for p in paths:
            unc_cas[state].append(ss.read_ca(p))
        print(f"uncoupled control {state}: {len(paths)} design(s) in {d}")

    rmsd_uncoupled = {"child_1": [], "child_2": []}
    # pair design i of the uncoupled parent run with design i of the
    # uncoupled child run - an arbitrary but fixed pairing (these runs are
    # otherwise unrelated; only the DISTRIBUTION of RMSDs across pairings
    # is meaningful, not any single pairing).
    n_pairs_1 = min(len(unc_cas["parent"]), len(unc_cas["child_1"]))
    n_pairs_2 = min(len(unc_cas["parent"]), len(unc_cas["child_2"]))
    for i in range(n_pairs_1):
        p1 = unc_cas["parent"][i][:split]
        c1 = unc_cas["child_1"][i]
        if m1.sum() >= 3 and len(p1) >= split and len(c1) >= split:
            rmsd_uncoupled["child_1"].append(kabsch_rmsd(c1[m1], p1[m1]))
    for i in range(n_pairs_2):
        p2 = unc_cas["parent"][i][split:]
        c2 = unc_cas["child_2"][i]
        if m2.sum() >= 3 and len(p2) >= (len(args.parent_dssp) - split) and len(c2) >= (len(args.parent_dssp) - split):
            rmsd_uncoupled["child_2"].append(kabsch_rmsd(c2[m2], p2[m2]))

    def summarize(name, xs):
        xs = np.asarray(xs, dtype=float)
        if xs.size == 0:
            return dict(n=0, mean=None, median=None)
        return dict(n=int(xs.size), mean=float(xs.mean()), median=float(np.median(xs)),
                   values=[float(x) for x in xs])

    report = dict(
        parent_len=len(args.parent_dssp), child_1_len=len(args.child_1_dssp),
        child_2_len=len(args.child_2_dssp), parent_split=split,
        n_tertiary_coupled=int(same_mask.sum()),
        psea_agreement=(dict((k, summarize(k, v)) for k, v in ss_results.items())),
        rmsd_coupled=(dict((k, summarize(k, v)) for k, v in rmsd_coupled.items())),
        rmsd_uncoupled_control=(dict((k, summarize(k, v)) for k, v in rmsd_uncoupled.items())),
    )
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n=== PSEA secondary-structure agreement (vs requested string, '_' skipped) ===")
    for state in ("parent", "child_1", "child_2"):
        s = report["psea_agreement"][state]
        print(f"  {state:>8}: n={s['n']}  mean_frac={s['mean']}")

    print("\n=== Tertiary-coupling RMSD at same_mask positions (Kabsch CA RMSD, Angstrom) ===")
    for state in ("child_1", "child_2"):
        c = report["rmsd_coupled"][state]
        u = report["rmsd_uncoupled_control"][state]
        print(f"  {state}: COUPLED   n={c['n']}  mean={c['mean']}  median={c['median']}")
        print(f"  {state}: UNCOUPLED n={u['n']}  mean={u['mean']}  median={u['median']}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
