"""R07: screw_replicate with L not divisible by count -- the trailing residues.

The port passes the remainder through UNCHANGED while re-centering the
symmetrised part, so the two halves drift apart over the sampling loop.
Upstream returns count*size residues instead (a shape error downstream).
"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
from salad_torch.inference.symmetry import screw_replicate
torch.manual_seed(0)

for L, n in ((101, 4), (100, 12), (60, 12), (40, 3)):
    size = L // n; rem = L - size * n
    x = torch.randn(L, 5, 3) * 8
    x0 = x.clone()
    drift = []
    for step in range(12):
        # the sampler's per-step Hook C: symmetrise, then feed back (plus a
        # small model-like perturbation so the loop is not a fixed point)
        x = screw_replicate(x, count=n, angle=2*np.pi/n, radius=0.0,
                            translation=0.0, chain_center=False, mean=False)
        core_c = x[:size*n, 1].mean(0)
        tail_c = x[size*n:, 1].mean(0) if rem else core_c
        drift.append(float((tail_c - core_c).norm()))
        x = x + torch.randn_like(x) * 0.05
    print(f"L={L:>4} count={n:>3} size={size:>3} remainder={rem:>2}  "
          f"tail-vs-core centroid separation over 12 steps: "
          f"{drift[0]:.2f} -> {drift[-1]:.2f} A")
