"""Summarise the per-step trajectory records written by x_traj.py."""
import json, sys, numpy as np
from collections import defaultdict

for path in sys.argv[1:]:
    try:
        rows = json.load(open(path))
    except Exception as e:
        print(f"### {path}: UNREADABLE ({e})"); continue
    by = defaultdict(list)
    for r in rows:
        by[r["design"]].append(r)
    print(f"\n################ {path}   designs={len(by)} steps/design={len(by[0])}")

    # --- SELF-CONDITIONING: is prev_pos at step N the output of step N-1? ---
    fed_back = reset = mismatch = 0
    first_reset = None
    for d, rs in by.items():
        rs.sort(key=lambda r: r["step"])
        for i, r in enumerate(rs):
            if r["prev_pos_norm"] == 0.0 and r["prev_local_norm"] == 0.0:
                reset += 1
                if first_reset is None or r["step"] < first_reset:
                    first_reset = r["step"]
            elif i > 0 and r["h_prev_in"] == rs[i - 1]["h_newprev"]:
                fed_back += 1
            elif i > 0:
                mismatch += 1
    print(f"  self-cond: fed-back {fed_back}, zero-reset {reset}, MISMATCH {mismatch}"
          f"  (first zero-reset at step {first_reset})")

    # --- RANDOMNESS: consecutive steps must draw different noise -----------
    dup_u = dup_n = 0
    for d, rs in by.items():
        us = [r["h_uniform"] for r in rs]
        ns = [r["h_noise"] for r in rs]
        dup_u += len(us) - len(set(us))
        dup_n += len(ns) - len(set(ns))
    print(f"  randomness: duplicate uniform hashes {dup_u}, duplicate noise hashes {dup_n} "
          f"(over {len(rows)} steps)")

    # --- WHERE does it go wrong ------------------------------------------
    print(f"  {'des':>4} {'final CA-CA':>11} {'final Rg':>9} {'1st>5A':>7} "
          f"{'1st>10A':>8} {'max jump x':>11} {'@step':>6} {'max|pos|':>10}")
    finals = []
    for d in sorted(by):
        rs = by[d]
        cc = np.array([r["ca_ca"] for r in rs])
        rg = np.array([r["rg"] for r in rs])
        pm = np.array([r["posmax"] for r in rs])
        f5 = np.argmax(cc > 5.0) if (cc > 5.0).any() else -1
        f10 = np.argmax(cc > 10.0) if (cc > 10.0).any() else -1
        ratio = cc[1:] / np.maximum(cc[:-1], 1e-9)
        j = int(np.nanargmax(ratio)); jump = float(ratio[j])
        finals.append((cc[-1], rg[-1]))
        print(f"  {d:>4} {cc[-1]:>11.3f} {rg[-1]:>9.2f} {f5:>7} {f10:>8} "
              f"{jump:>11.2f} {j+1:>6} {np.nanmax(pm):>10.1f}")
    fc = np.array([f[0] for f in finals]); fr = np.array([f[1] for f in finals])
    ok = (fc < 4.5) & (fc > 3.0) & (fr < 30)
    print(f"  SUMMARY diverged {int((~ok).sum())}/{len(ok)}   converged CA-CA "
          f"{fc[ok].mean() if ok.any() else float('nan'):.4f}  Rg "
          f"{fr[ok].mean() if ok.any() else float('nan'):.2f}")

    # --- per-step means over CONVERGED designs only -----------------------
    good = [d for d in sorted(by) if ok[list(sorted(by)).index(d)]]
    if good:
        n = len(by[good[0]])
        cc = np.array([[by[d][i]["ca_ca"] for i in range(n)] for d in good])
        rgm = np.array([[by[d][i]["rg"] for i in range(n)] for d in good])
        print("  per-step (converged only)  step: CA-CA / Rg")
        for s in list(range(0, n, max(1, n // 12))) + [n - 1]:
            print(f"     {s:>4}: {np.nanmean(cc[:, s]):.3f} / {np.nanmean(rgm[:, s]):.2f}")
