"""CA-trace secondary structure + backbone sanity, for the N011 quality gate.

No DSSP dependency (which would need hydrogens and a full backbone): this is
the P-SEA criterion (Labesse et al. 1997), which assigns helix/strand from the
CA trace alone using the i->i+2/3/4 distances and the CA pseudo-bond angle and
pseudo-torsion. It is the standard CA-only assignment and is what a generated
backbone can actually be scored with.

  helix   d(i,i+2) 5.5+-0.5  d(i,i+3) 5.3+-0.5  d(i,i+4) 6.4+-0.6
          theta 89+-12 deg   alpha 50+-20 deg
  strand  d(i,i+2) 6.7+-0.6  d(i,i+3) 9.9+-0.9  d(i,i+4) 12.4+-1.1
          theta 124+-14 deg  alpha -170+-45 deg (or +170)

Usage:  python ss.py <dir-with-pdbs> [...]
prints one line per directory: n, CA-CA mean/min/max, Rg, helix/strand/coil %,
clash count, and NaN count.
"""
from __future__ import annotations

import glob, math, os, sys
import numpy as np


def read_ca(path):
    ca = []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(ca, dtype=np.float64)


def _ang(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _tors(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def psea(ca):
    """Returns a per-residue string of 'H', 'E', 'c'."""
    n = len(ca)
    ss = np.array(["c"] * n)
    if n < 6:
        return "".join(ss)
    d = lambda k: np.linalg.norm(ca[k:] - ca[:-k], axis=-1)   # noqa: E731
    d2, d3, d4 = d(2), d(3), d(4)
    theta = _ang(ca[:-2], ca[1:-1], ca[2:])
    alpha = _tors(ca[:-3], ca[1:-2], ca[2:-1], ca[3:])

    def band(x, m, s):
        return np.abs(x - m) <= s

    # a residue i is the START of a 5-residue helix/strand window
    helix = np.zeros(n, bool)
    strand = np.zeros(n, bool)
    for i in range(n - 4):
        h_d = (band(d2[i], 5.5, .5) and band(d2[i + 1], 5.5, .5)
               and band(d3[i], 5.3, .5) and band(d3[i + 1], 5.3, .5)
               and band(d4[i], 6.4, .6))
        h_a = (band(theta[i], 89, 12) and band(theta[i + 1], 89, 12)
               and band(alpha[i], 50, 20))
        if h_d or h_a:
            helix[i:i + 5] = True
    for i in range(n - 4):
        e_d = (band(d2[i], 6.7, .6) and band(d2[i + 1], 6.7, .6)
               and band(d3[i], 9.9, .9) and band(d3[i + 1], 9.9, .9)
               and band(d4[i], 12.4, 1.1))
        a = alpha[i]
        e_a = (band(theta[i], 124, 14) and band(theta[i + 1], 124, 14)
               and (band(a, 180, 45) or band(a, -180, 45)))
        if e_d or e_a:
            strand[i:i + 5] = True
    ss[helix] = "H"
    ss[strand & ~helix] = "E"
    return "".join(ss)


def summarise(paths):
    rows = []
    for p in paths:
        ca = read_ca(p)
        if len(ca) < 3:
            continue
        dca = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
        rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))
        s = psea(ca)
        dm = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
        iu = np.triu_indices(len(ca), 3)
        rows.append(dict(
            path=p, n=len(ca),
            nan=int(np.isnan(ca).sum()),
            ca_mean=float(dca.mean()), ca_min=float(dca.min()), ca_max=float(dca.max()),
            rg=rg,
            helix=s.count("H") / len(s), strand=s.count("E") / len(s),
            coil=s.count("c") / len(s),
            clash=int((dm[iu] < 4.0).sum()),
        ))
    return rows


if __name__ == "__main__":
    for d in sys.argv[1:]:
        paths = sorted(glob.glob(os.path.join(d, "*.pdb")))
        rows = summarise(paths)
        if not rows:
            print(f"{d}: no pdbs")
            continue
        f = lambda k: np.array([r[k] for r in rows])   # noqa: E731
        print(f"== {d}   n_designs={len(rows)}  L={rows[0]['n']}")
        for r in rows:
            print(f"   {os.path.basename(r['path']):<24} CA-CA {r['ca_mean']:.3f} "
                  f"(min {r['ca_min']:.2f} max {r['ca_max']:.2f})  Rg {r['rg']:5.1f}  "
                  f"H {r['helix']:.2f} E {r['strand']:.2f} c {r['coil']:.2f}  "
                  f"clash<4A {r['clash']:3d}  nan {r['nan']}")
        print(f"   MEAN  CA-CA {f('ca_mean').mean():.4f} +- {f('ca_mean').std():.4f}  "
              f"Rg {f('rg').mean():.2f} +- {f('rg').std():.2f}  "
              f"H {f('helix').mean():.3f}  E {f('strand').mean():.3f}  "
              f"c {f('coil').mean():.3f}  clash {f('clash').mean():.1f}  "
              f"nan {int(f('nan').sum())}")
