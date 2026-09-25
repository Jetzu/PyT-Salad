"""R01: clash_repulsion_grad -- torch port vs JAX reference.

JAX  : dist = sqrt( sum_c max(dx_c, 1e-3)**2 )      <- clamps the DIFFERENCE
Torch: dist = sqrt( sum_c max(dx_c**2, 1e-3) )      <- clamps the SQUARE
"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
from salad_torch.inference.symmetry import clash_repulsion_grad

rng = np.random.default_rng(0)
ca = rng.normal(scale=6.0, size=(24, 3)).astype(np.float32)
np.save("/tmp/r01_ca.npy", ca)
g = clash_repulsion_grad(torch.from_numpy(ca)).numpy()
np.save("/tmp/r01_torch.npy", g)
print("torch grad norm", np.linalg.norm(g), "max", np.abs(g).max())
