"""N029 - paired n=160 comparison of two test_design_quality logs (N015's method)."""
import re, sys
import numpy as np
from scipy import stats

def load(p):
    d = {}
    for l in open(p):
        m = re.match(r"(design_\d+)\.pdb\s+(\d+)\s+([\d.]+)\s+([\d.]+)", l.strip())
        if m:
            d[m.group(1)] = (float(m.group(3)), float(m.group(4)))
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) & set(b))
print(f"paired designs: {len(keys)} (A {len(a)}, B {len(b)})")
for i, name in enumerate(("CA-CA", "Rg")):
    x = np.array([a[k][i] for k in keys]); y = np.array([b[k][i] for k in keys])
    t, pt = stats.ttest_ind(x, y, equal_var=False)
    ks, pk = stats.ks_2samp(x, y)
    bf, pb = stats.levene(x, y, center="median")
    print(f"{name:6s} A mean {x.mean():.4f} sd {x.std(ddof=1):.4f} | "
          f"B mean {y.mean():.4f} sd {y.std(ddof=1):.4f} | "
          f"Welch p={pt:.3f}  KS p={pk:.3f}  Brown-Forsythe p={pb:.3f}")
    if name == "Rg":
        print(f"       Rg>18 outliers: A {(x>18).sum()}/{len(x)}  B {(y>18).sum()}/{len(y)}")
