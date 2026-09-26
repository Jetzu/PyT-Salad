"""Turn the `CELL ...` lines of the surface jobs into the trade-off table.

Usage: python parse_surface.py logs/surf_*.log
"""
import re
import sys
from collections import defaultdict

rows = []
oom = []
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("CELL "):
            d = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
            for k, v in list(d.items()):
                try:
                    d[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                except ValueError:
                    pass
            rows.append(d)
        elif line.startswith(("CELL_OOM", "CELL_FAIL", "CELL_TIMEOUT")):
            oom.append(line.split("  ")[0][:110])

by = {}
for r in rows:
    by[(r["L"], r["B"], r["G"])] = r

Ls = sorted({r["L"] for r in rows})
print(f"{'L':>5} {'B':>3} {'G':>3} {'B*G':>4} {'replay_lean_ms':>15} "
      f"{'host_ms/step':>13} {'designs/s':>10} {'s/design':>9} "
      f"{'peakVRAM_MiB':>13} {'MiB/design':>11} {'vs B=1,G=1':>11}")
for L in Ls:
    base = None
    for (l, b, g), r in sorted(by.items()):
        if l == L and b == 1 and g == 1:
            base = r["designs_per_s"]
    print("-" * 118)
    best = None
    for (l, b, g), r in sorted(by.items()):
        if l != L:
            continue
        spd = r["designs_per_s"] / base if base else float("nan")
        if best is None or r["designs_per_s"] > best[0]:
            best = (r["designs_per_s"], b, g)
        print(f"{l:>5} {b:>3} {g:>3} {b*g:>4} {r['replay_lean_ms']:>15.3f} "
              f"{r['host_step_ms']:>13.3f} {r['designs_per_s']:>10.3f} "
              f"{r['sec_per_design']:>9.4f} {r['peak_reserved_MiB']:>13.1f} "
              f"{r['peak_reserved_MiB_per_design']:>11.1f} {spd:>10.2f}x")
    if best:
        print(f"  -> L={L} BEST: B={best[1]} G={best[2]} "
              f"{best[0]:.3f} designs/s "
              f"({best[0]/base:.2f}x the B=1,G=1 baseline)" if base else "")
if oom:
    print("\nCELLS THAT DID NOT PRODUCE A NUMBER:")
    for o in sorted(set(oom)):
        print("  " + o)
