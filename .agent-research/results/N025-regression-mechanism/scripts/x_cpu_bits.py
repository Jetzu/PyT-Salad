"""Bitwise CPU comparison of `forward_core` between two commits.

CPU is deterministic by construction, so any difference here is the MODEL CODE,
not compilation, not TF32, not atomics. Inputs are generated from a fixed seed
and DUMPED, so both commits are fed the identical bits.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--num_aa", type=int, default=64)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--inputs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--params", default="/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
ap.add_argument("--config", default="default_vp")
a = ap.parse_args()

sys.path.insert(0, a.root)
from salad_torch import opt as salad_opt                                     # noqa
from salad_torch.config import get_config                                    # noqa
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz    # noqa
from salad_torch.modules.diffusion import uniform_draw_count                 # noqa
from salad_torch.modules.model import StructureDiffusionInference            # noqa

L, A = a.num_aa, 20
cfg = get_config(a.config)
nd = uniform_draw_count(cfg)
if not os.path.exists(a.inputs):
    g = torch.Generator().manual_seed(a.seed)
    d = dict(
        pos_noised=torch.randn(L, A, 3, generator=g) * 5,
        prev_pos=torch.randn(L, A, 3, generator=g) * 5,
        prev_local=torch.randn(L, 128, generator=g),
        condition=torch.randn(L, 128, generator=g),
        dmap=torch.zeros(L, L), omap=torch.zeros(L, L, 9),
        dmap_mask=torch.zeros(L, L, dtype=torch.bool), flags=torch.zeros(L, L, 3),
        t_pos=torch.full((L,), 0.5), t_seq=torch.ones(L),
        seq_noised=torch.full((L,), 20, dtype=torch.long),
        aa_gt=torch.full((L,), 20, dtype=torch.long),
        pos_gt=torch.randn(L, A, 3, generator=g) * 5,
        resi=torch.arange(L), chain=torch.zeros(L, dtype=torch.long),
        batch=torch.zeros(L, dtype=torch.long), mask=torch.ones(L, dtype=torch.bool),
        uniform=torch.rand(nd, L, L, generator=g).clamp_(1e-6, 1 - 1e-6),
    )
    torch.save(d, a.inputs)
    print("[x] wrote inputs", a.inputs, flush=True)
d = torch.load(a.inputs)

model = StructureDiffusionInference(cfg).eval()
load_model(model.predict, load_param_npz(a.params), cfg, strict=True)
order = ("pos_noised", "prev_pos", "prev_local", "condition", "dmap", "omap",
         "dmap_mask", "flags", "t_pos", "t_seq", "seq_noised", "aa_gt",
         "pos_gt", "resi", "chain", "batch", "mask")
args = [d[k] for k in order] + [None, d["uniform"]]   # cyclic_mask, uniform
print("[x] BATCH_COND =", getattr(salad_opt, "BATCH_COND", "absent"),
      " DET_SEGMENT =", getattr(salad_opt, "DET_SEGMENT", "absent"),
      " HOIST =", salad_opt.HOIST_NEIGHBOURS, " NOPC =", salad_opt.SKIP_PAIR_COND,
      flush=True)
with torch.no_grad():
    out = model.predict.diffusion.forward_core(*args, True, True, False)
res = {}
names = ("pos", "local", "trajectory", "aa", "aa_features", "dssp",
         "distogram_neighbours", "raw_angles", "angles", "angle_pos", "aatype")
for n, o in zip(names, out):
    if torch.is_tensor(o):
        res[n] = o.detach().cpu().numpy()
np.savez(a.out, **res)
print("[x] wrote", a.out, {k: (v.shape, float(np.abs(v).max())) for k, v in
                           list(res.items())[:3]}, flush=True)
