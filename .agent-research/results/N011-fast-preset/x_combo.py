"""N011 - can the fast preset unblock inductor's horizontal fusion?

`torch._inductor.config.combo_kernels = True` merges independent kernels into
one launch. It is exactly the right lever for a model whose floor is ~946
kernels at ~2-3 us each and whose arithmetic floor is 0.21 ms — it is the only
setting that removes launches by the hundred without hand-writing Triton.

N010 reports that it CRASHES codegen, on the stable-argsort node. The exact
path is stuck there: that argsort is the neighbour selector's contract with
JAX. The fast preset is not — it is allowed to select neighbours differently.
So the hypothesis worth one job is:

    topk (measured +0.19 ms on its own, i.e. a real cost) may PAY FOR ITSELF
    several times over if removing the argsort node is what lets combo_kernels
    compile.

Variants:
  combo            combo_kernels on, argsort untouched  -> expect the crash
  combo_topk       combo_kernels on, argsort -> topk    -> the hypothesis
  topk             topk alone, no combo                 -> the control
  base             neither                              -> the baseline

All four run the production `FastDenoiser(preset="fast")`, one per process.
"""
from __future__ import annotations

import argparse, sys, time
import torch

sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")

from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.geometry import neighbours as NB
from salad_torch.modules import blocks as BL
from salad_torch.modules.fast import FastDenoiser
from salad_torch.modules.model import StructureDiffusionPredict

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="base")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--preset", default="fast")
ap.add_argument("--iters", type=int, default=30)
a = ap.parse_args()


def patch_topk():
    """Stage selection by `topk`. Same selected distances, unspecified tie
    order (argsort(stable=True) breaks ties by ascending index; topk does not
    say). `min(count, n)` reproduces argsort+slice's behaviour on a row
    shorter than `count`."""
    def get_neighbours(count: int):
        def inner(distance, mask, neighbours=None):
            n = distance.shape[0]
            index = torch.arange(n, device=distance.device)
            inf = float("inf")
            distance = torch.where(mask.bool(), distance,
                                   torch.full_like(distance, inf))
            if neighbours is not None:
                update = torch.where(
                    neighbours != -1,
                    torch.full_like(distance[index[:, None], neighbours], inf),
                    distance[index[:, None], neighbours])
                distance = distance.clone()
                distance[index[:, None], neighbours] = update
            k = min(count, n)
            if k <= 0:
                knn = torch.zeros((n, 0), dtype=torch.long, device=distance.device)
            else:
                picked, knn = torch.topk(distance, k, dim=-1, largest=False,
                                         sorted=True)
                knn = torch.where(picked < inf, knn, torch.full_like(knn, -1))
            if neighbours is not None:
                knn = torch.cat((neighbours, knn), dim=-1)
            return knn
        return inner
    NB.get_neighbours = get_neighbours
    BL.get_neighbours = get_neighbours


combo = a.variant in ("combo", "combo_topk")
if a.variant in ("topk", "combo_topk"):
    patch_topk()
if combo:
    import torch._inductor.config as ic
    ic.combo_kernels = True
    # benchmark each merged kernel against the separate ones and keep the
    # merge only when it wins; without this inductor merges unconditionally
    if hasattr(ic, "benchmark_combo_kernel"):
        ic.benchmark_combo_kernel = True

dev = torch.device("cuda")
cfg = get_config("default_vp")
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"),
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


t0 = time.perf_counter()
with torch.no_grad():
    fast = FastDenoiser(m, L, device=dev, preset=a.preset).warmup(pair_cond=False)
    t_compile = time.perf_counter() - t0
    t_full = bench(lambda: fast(d, p), a.iters)
    t_lean = bench(lambda: fast(d, p, need_aa=False), a.iters)
print(f"RESULT combo variant={a.variant:<11} preset={a.preset:<5} L={L:<4} "
      f"full {t_full:6.3f} ms  lean {t_lean:6.3f} ms  compile {t_compile:5.0f}s",
      flush=True)
