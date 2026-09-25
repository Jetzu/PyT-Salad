"""R02: screw_replicate torch vs JAX over a parameter grid."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
from salad_torch.inference.symmetry import screw_replicate
rng = np.random.default_rng(7)
CASES = [  # (L, count, angle_deg, radius, translation, chain_center, mean, time)
    ("c4 L=100 mean=F",      100, 4, 90.0, 0.0, 0.0, False, False, 0.0),
    ("c4 L=100 mean=T",      100, 4, 90.0, 0.0, 0.0, False, True,  0.0),
    ("c4 L=100 cc=T t=.9",   100, 4, 90.0, 0.0, 0.0, True,  False, 0.9),
    ("c4 L=101 (indiv)",     101, 4, 90.0, 0.0, 0.0, False, False, 0.0),
    ("c1 L=40",               40, 1,360.0, 0.0, 0.0, False, False, 0.0),
    ("c1 L=40 cc=T t=.9",     40, 1,360.0, 0.0, 0.0, True,  False, 0.9),
    ("c1 L=40 radius=15",     40, 1,360.0,15.0, 0.0, False, False, 0.0),
    ("c12 L=96 screw",        96,12, 30.0,15.0,15.0, False, False, 0.0),
    ("c12 L=100 (indiv)",    100,12, 30.0,15.0,15.0, False, False, 0.0),
    ("c2 L=40 r=0 tr=1e6",    40, 2,180.0, 0.0,1e6,  False, False, 0.0),
]
out = {}
for name, L, cnt, ang, rad, tr, cc, mn, tm in CASES:
    x = rng.normal(scale=8.0, size=(L, 5, 3)).astype(np.float32)
    y = screw_replicate(torch.from_numpy(x), count=cnt, angle=ang/180*np.pi,
                        radius=rad, translation=tr, chain_center=cc, mean=mn, time=tm)
    out[name + "|x"] = x
    out[name + "|y"] = y.numpy()
    out[name + "|p"] = np.array([L, cnt, ang, rad, tr, float(cc), float(mn), tm])
np.savez("/tmp/r02.npz", **out)
print("wrote", len(CASES), "cases")
