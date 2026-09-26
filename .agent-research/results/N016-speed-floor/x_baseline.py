"""N016 - re-measure the baseline and SPLIT it into graph-replay vs wrapper.

The headline 3.554 / 2.894 ms are `FastDenoiser.__call__` wall times, i.e.
they include the python wrapper (input copies into the static buffers, the
conditioning-cache hash, output clones). A floor argument has to know how much
of the number is the captured graph and how much is host-side, because the two
have completely different floors.

Prints:
  BASE  wall time of the production call, full + lean
  REPLAY  the captured graph's own replay time (CUDA events, no wrapper)
  WRAP    the difference
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

ROOT = "/hnvme/workspace/v181bb10-salad/n016-floor"
sys.path.insert(0, ROOT)

from salad_torch.config import get_config                      # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.fast import FastDenoiser, PRESETS      # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="exact")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--iters", type=int, default=200)
a = ap.parse_args()

dev = torch.device("cuda")
cfg = get_config(a.config)
L, A = a.length, 5 + cfg.augment_size
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"),
           cfg, strict=True)

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


def wall(fn, it, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e3


def ev(fn, it, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it


with torch.no_grad():
    f = FastDenoiser(m, L, device=dev, preset=a.preset)
    f.warmup(pair_cond=False)
    t_full = wall(lambda: f(d, p), a.iters)
    t_lean = wall(lambda: f(d, p, need_aa=False), a.iters) if f.lean_steps else t_full
    # the captured graphs, replayed directly
    reps = {}
    for key, (g, _outs) in f._graphs.items():
        reps[key] = (ev(g.replay, a.iters), wall(g.replay, a.iters))
print(f"BASE preset={a.preset} L={L} full={t_full:.4f} ms lean={t_lean:.4f} ms "
      f"tf32={torch.get_float32_matmul_precision()}", flush=True)
for key, (e_ms, w_ms) in sorted(reps.items()):
    print(f"REPLAY preset={a.preset} (need_aa,pair_cond)={key} "
          f"event={e_ms:.4f} ms wall={w_ms:.4f} ms", flush=True)
lean_key = (False, False)
if lean_key in reps:
    print(f"WRAP preset={a.preset} lean wrapper = {t_lean - reps[lean_key][0]:.4f} ms "
          f"({(t_lean - reps[lean_key][0]) / t_lean * 100:.1f}% of the step)", flush=True)
