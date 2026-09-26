"""N021 - amino-acid composition per arm, as a second discrete signal.

The controlled per-residue aatype divergence lives in `x_divergence.py` (same
randomness, one design at two offsets). This is the population-level version:
does the packed sampler emit a different distribution of amino acids?
Chi-square over the 20x(arms) contingency table, plus per-arm frequencies.
"""
import glob
import os
import sys

import numpy as np
from scipy import stats as sps

AA3 = ("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR "
       "TRP TYR VAL").split()


def composition(d):
    counts = np.zeros(20, dtype=np.int64)
    for p in sorted(glob.glob(os.path.join(d, "*.pdb"))):
        for line in open(p):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                r = line[17:20]
                if r in AA3:
                    counts[AA3.index(r)] += 1
    return counts


dirs = sys.argv[1:]
names = [os.path.basename(x.rstrip("/")) for x in dirs]
table = np.stack([composition(d) for d in dirs])
print(f"{'aa':>4} " + " ".join(f"{n:>10}" for n in names))
for i, aa in enumerate(AA3):
    print(f"{aa:>4} " + " ".join(f"{table[j, i] / table[j].sum():>10.4f}"
                                for j in range(len(dirs))))
print(f"{'N':>4} " + " ".join(f"{int(table[j].sum()):>10}" for j in range(len(dirs))))
chi2, p, dof, _ = sps.chi2_contingency(table)
print(f"\nchi-square over the whole table: chi2={chi2:.2f} dof={dof} p={p:.4f}")
for j in range(1, len(dirs)):
    c2, pj, d2, _ = sps.chi2_contingency(table[[0, j]])
    print(f"  {names[j]} vs {names[0]}: chi2={c2:.2f} dof={d2} p={pj:.4f}"
          + ("  **" if pj < 0.01 else ("  *" if pj < 0.05 else "")))
