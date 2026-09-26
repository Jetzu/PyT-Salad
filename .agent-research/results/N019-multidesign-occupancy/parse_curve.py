"""Collect every `CURVE ...` line from the sweep logs into two tables."""
from __future__ import annotations

import glob
import sys

rows = []
for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                             else "logs/curve_*.log")):
    for line in open(path):
        if not line.startswith("CURVE "):
            continue
        kv = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
        kv["_log"] = path
        rows.append(kv)

def f(r, k, d=float("nan")):
    try:
        return float(r[k])
    except Exception:
        return d

presets = sorted({r["preset"] for r in rows})
for pre in presets:
    lens = sorted({int(r["L"]) for r in rows if r["preset"] == pre})
    for L in lens:
        sel = [r for r in rows if r["preset"] == pre and int(r["L"]) == L]
        base = next((r for r in sel if r["requested"] == "1"), None)
        if base is None:
            continue
        b_lean = f(base, "replay_lean_per_design_ms")
        b_host = f(base, "host_step_per_design_ms")
        print(f"\n### preset={pre} L={L}   (N=1 baseline: "
              f"replay {b_lean:.3f} ms/design, host-inclusive step "
              f"{b_host:.3f} ms/design)")
        print(f"{'N':>4} {'req':>5} {'replay/des ms':>13} {'x':>6} "
              f"{'hoststep/des ms':>15} {'x':>6} {'peak MiB':>9} "
              f"{'MiB/design':>10}")
        for r in sorted(sel, key=lambda r: (int(r["N"]), r["requested"])):
            n = int(r["N"])
            lp, hp = f(r, "replay_lean_per_design_ms"), f(r, "host_step_per_design_ms")
            pk = f(r, "peak_reserved_MiB")
            print(f"{n:>4} {r['requested']:>5} {lp:>13.4f} {b_lean/lp:>6.2f} "
                  f"{hp:>15.4f} {b_host/hp:>6.2f} {pk:>9.1f} {pk/n:>10.1f}")
