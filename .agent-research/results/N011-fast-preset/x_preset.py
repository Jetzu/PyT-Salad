"""N011 - benchmark the PRODUCTION `FastDenoiser` preset, one config per process.

Unlike `x_fast.py` (which monkeypatches a private copy of the mechanism to
explore candidates) this runs exactly what `scripts/generate.py --fast
--preset X` runs, so its numbers are the ones the preset actually delivers.

`--override` flips one switch away from the preset's choice, between
`__init__` (which applies the preset) and `warmup()` (which compiles), so a
single switch's contribution can be isolated without a second preset:

    --override no_shared_draw   fastopt.SHARED_COND_DRAW = False
    --override shared_draw      fastopt.SHARED_COND_DRAW = True
    --override tf32             float32 matmul precision "high"
    --override notf32           float32 matmul precision "highest"

Usage:  python x_preset.py --preset fast --length 100
"""
from __future__ import annotations

import argparse, sys, time
import torch

sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")

from salad_torch import fastopt, opt
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.fast import PRESETS, FastDenoiser
from salad_torch.modules.model import StructureDiffusionPredict

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="exact")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--override", default="")
a = ap.parse_args()

dev = torch.device("cuda")
cfg = get_config(a.config)
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"),
           cfg, strict=True)

L, A = a.length, 5 + cfg.augment_size
z = torch.zeros(L, dtype=torch.long, device=dev)
pos = torch.randn(L, A, 3, device=dev) * 4
d = dict(pos=pos, pos_noised=pos,
         aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
         residue_index=torch.arange(L, device=dev), chain_index=z, batch_index=z,
         mask=torch.ones(L, dtype=torch.bool, device=dev),
         t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev))
p = {"pos": torch.zeros(L, A, 3, device=dev),
     "local": torch.zeros(L, cfg.local_size, device=dev)}


def bench(fn, it, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e3


with torch.no_grad():
    fast = FastDenoiser(m, L, device=dev, preset=a.preset)
    # the preset has been applied; override one switch before compiling
    if a.override == "no_shared_draw":
        fastopt.configure(shared_cond_draw=False)
    elif a.override == "shared_draw":
        fastopt.configure(shared_cond_draw=True)
    elif a.override == "tf32":
        torch.set_float32_matmul_precision("high")
    elif a.override == "notf32":
        torch.set_float32_matmul_precision("highest")
    elif a.override:
        raise SystemExit(f"unknown override {a.override!r}")
    fast.warmup(pair_cond=False)
    t_full = bench(lambda: fast(d, p), a.iters)
    t_lean = (bench(lambda: fast(d, p, need_aa=False), a.iters)
              if fast.lean_steps else t_full)

sc = float(PRESETS[a.preset].get("steps_scale", 1.0))
steps = max(1, int(round(400 * sc)))     # generate.py stops at out_steps[-1]
design_s = (t_full + steps * t_lean) / 1000
print(f"RESULT preset={a.preset:<6} override={a.override or '-':<15} L={L:<4} "
      f"full {t_full:6.3f} ms  lean {t_lean:6.3f} ms  "
      f"tf32={torch.get_float32_matmul_precision()} {fastopt.state()} "
      f"trust={fast.trust_cond_cache} steps={steps + 1} "
      f"design={design_s:5.2f} s", flush=True)
