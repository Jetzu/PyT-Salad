import sys
import numpy as np
A, B, label = sys.argv[1], sys.argv[2], sys.argv[3]
a, b = np.load(A), np.load(B)
worst = []
for k in a.files:
    if k not in b.files or a[k].shape != b[k].shape:
        print(f"  {k}: MISSING/SHAPE"); continue
    x, y = a[k].astype(np.float64), b[k].astype(np.float64)
    d = float(np.abs(x - y).max())
    worst.append((k, d))
p = dict(worst).get("pos", float("nan"))
dn = dict(worst).get("distogram_neighbours", float("nan"))
print(f"CMP {label:34} pos={p:.6e}  distnbr={dn:.6e}  "
      + " ".join(f"{k}={d:.2e}" for k, d in worst if k not in ("pos",)), flush=True)
