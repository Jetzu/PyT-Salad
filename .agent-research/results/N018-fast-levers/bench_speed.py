"""N018 - benchmark the PRODUCTION `FastDenoiser` preset, one config per process.

Adapted from N011's x_preset.py (.agent-research/results/N011-fast-preset/):
same production entry point (`FastDenoiser(preset=...)`, not a monkeypatched
private copy), same bench() protocol. Points at the N018 WORKTREE
(/hnvme/workspace/v181bb10-salad/n018-worktree), which carries two new preset
entries not in the shared tree: "fast_halfnbr" (N011's K64->32 patch,
reproduced) and "fast_randhalf" (random-neighbours-only halved). Both set
steps_scale=1.0 so the printed ms figures measure the KERNEL, never the step
count - the brief's explicit requirement for all speed comparisons.

Runs `reps` independent bench() rounds after one warmup/capture, so a variance
estimate does not need a full recompile per rep.

Usage:  python bench_speed.py --preset fast_halfnbr --length 100 --reps 3
"""
from __future__ import annotations

import argparse, sys, time
import torch

ROOT = "/hnvme/workspace/v181bb10-salad/n018-worktree"
sys.path.insert(0, ROOT)

from salad_torch import opt  # noqa: E402
from salad_torch.config import get_config  # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.fast import PRESETS, FastDenoiser  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="exact")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--reps", type=int, default=3)
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
    fast.warmup(pair_cond=False)
    full_reps = [bench(lambda: fast(d, p), a.iters) for _ in range(a.reps)]
    lean_reps = ([bench(lambda: fast(d, p, need_aa=False), a.iters)
                 for _ in range(a.reps)] if fast.lean_steps else full_reps)

import statistics as st  # noqa: E402
fm, fs = st.mean(full_reps), (st.stdev(full_reps) if len(full_reps) > 1 else 0.0)
lm, ls_ = st.mean(lean_reps), (st.stdev(lean_reps) if len(lean_reps) > 1 else 0.0)
print(f"RESULT preset={a.preset:<14} L={L:<4} nbr_mode={opt.NBR_MODE:<9} "
      f"tf32={torch.get_float32_matmul_precision()} "
      f"full {fm:6.3f}+-{fs:.3f} ms  lean {lm:6.3f}+-{ls_:.3f} ms  "
      f"reps={a.reps} raw_full={['%.3f' % x for x in full_reps]} "
      f"raw_lean={['%.3f' % x for x in lean_reps]}", flush=True)
