"""Diff two x_identity.py output directories, design for design."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

a_dir, b_dir = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else ""
names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a_dir, "*.npz")))
if not names:
    print(f"CMP {label}: NO FILES in {a_dir}")
    raise SystemExit(1)
worst = 0.0
worst_name = ""
bitex = 0
aadiff_total = 0
n_res_total = 0
for nm in names:
    A = np.load(os.path.join(a_dir, nm))
    bp = os.path.join(b_dir, nm)
    if not os.path.exists(bp):
        print(f"CMP {label}: MISSING {nm} in {b_dir}")
        raise SystemExit(1)
    B = np.load(bp)
    pa, pb = A["pos"], B["pos"]
    d = float(np.abs(pa - pb).max())
    same_bits = bool((pa.tobytes() == pb.tobytes()))
    bitex += int(same_bits)
    aa_a, aa_b = A["aatype"], B["aatype"]
    nd = int((aa_a != aa_b).sum()) if aa_a.shape == aa_b.shape else -1
    aadiff_total += max(nd, 0)
    n_res_total += aa_a.size
    if d > worst:
        worst, worst_name = d, nm
    print(f"CMP {label} {nm}: max|dpos|={d:.3e} A  bit_identical={same_bits}  "
          f"aatype_diff={nd}/{aa_a.size}")
print(f"CMP_SUMMARY {label}: n={len(names)} worst={worst:.6e} A ({worst_name}) "
      f"bit_identical={bitex}/{len(names)} aatype_diff={aadiff_total}/{n_res_total}")
