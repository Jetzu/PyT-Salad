"""R06: measure the actual Cn symmetry residual on generate.py's written PDBs."""
import glob, math, os, sys
import numpy as np
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")

def read_ca(path):
    ca, ch = [], []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            ch.append(line[21])
    return np.array(ca), np.array(ch)

def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, W = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ W))
    R = V @ np.diag([1, 1, d]) @ W
    return float(np.sqrt((((Pc @ R) - Qc) ** 2).sum(1).mean()))

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float64)

CASES = [("c4_div", 4), ("c4_nondiv", 4), ("c12_small", 12), ("c12_L100", 12),
         ("screw_zero_r", 4), ("sym_mean", 4), ("thr_hi", 4), ("rc_gt_L", 16), ("c1", 1)]
print(f"{'case':<14}{'L':>5}{'n':>4}{'size':>6}{'rem':>5}  {'exact-Cn rmsd':>14} {'free-superpos':>14}  {'max CA-CA':>10}")
for tag, n in CASES:
    p = f"/tmp/rt_gen/{tag}/design_0.pdb"
    if not os.path.exists(p): print(tag, "missing"); continue
    ca, ch = read_ca(p)
    L = len(ca); size = L // max(n, 1); rem = L - size * n
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    if size == 0 or n <= 1:
        print(f"{tag:<14}{L:>5}{n:>4}{size:>6}{rem:>5}  {'n/a':>14} {'n/a':>14}  {d.max():>10.2f}")
        continue
    used = ca[:size * n]
    c = used.mean(0)
    # exact Cn test: does rotating subunit 0 by 2pi*i/n reproduce subunit i?
    exact, free = [], []
    s0 = used[:size] - c
    for i in range(1, n):
        si = used[i*size:(i+1)*size] - c
        R = rot_y(2 * math.pi * i / n)
        exact.append(float(np.sqrt((((s0 @ R.T) - si) ** 2).sum(1).mean())))
        free.append(kabsch_rmsd(used[:size], used[i*size:(i+1)*size]))
    print(f"{tag:<14}{L:>5}{n:>4}{size:>6}{rem:>5}  {max(exact):>14.4f} {max(free):>14.4f}  {d.max():>10.2f}")
    if rem:
        tail = ca[size*n:]
        dmin = np.linalg.norm(tail[:, None] - used[None], axis=-1).min()
        print(f"{'':<14}  -> {rem} un-symmetrised trailing residue(s); "
              f"nearest CA to the symmetric core: {dmin:.2f} A; "
              f"CA-CA across the seam: {np.linalg.norm(ca[size*n]-ca[size*n-1]):.2f} A")
