"""N021 - what fraction of the roofline does packed batching actually deliver?

N016's rooflines on this node: fp32 FMA peak 66.91 TFLOP/s, L2 6.53 TB/s, DRAM
2.94 TB/s, and one lean L=100 forward is 10.014 GFLOP of matmul.

Two arithmetic rates are reported and the difference is the whole point:

* EXECUTED  = FLOPs the packed forward really issues / replay time. Packing
  raises this because the same launch does more work.
* USEFUL    = B x (FLOPs of a SINGLE-design forward at the same L) / replay
  time. Any gap between the two is work packing ADDED - salad's neighbour
  selection, its Gumbel field and its pair maps are all O(N^2) over the FLAT
  residue axis, so they grow as B^2 while the designs grow as B.

`--tf32` reports against the vendor TF32 dense number as well, but the honest
comparison for the `fast` preset is the measured 77.2 TFLOP/s N016 got out of
a real TF32 GEMM here (its R6 caveat).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n021tree")
sys.path.insert(0, ROOT)

from salad_torch.config import get_config                                # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.diffusion.sampling import make_initial_data, pack_designs  # noqa: E402
from salad_torch.diffusion.schedules import parse_num_aa                 # noqa: E402
from salad_torch.modules.fast import FastDenoiser, GroupFastDenoiser     # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="fast")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--batch_size", type=int, default=1)
ap.add_argument("--group_size", type=int, default=1)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--reps", type=int, default=40)
a = ap.parse_args()

PEAK_FP32 = 66.91e12
PEAK_TF32_VENDOR = 494.7e12
PEAK_TF32_MEASURED = 77.2e12

dev = torch.device("cuda")
cfg = get_config(a.config)
L, B, G = a.length, a.batch_size, a.group_size
model = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(model, load_param_npz(
    f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"), cfg, strict=True)


def flops_for(batch_size: int) -> float:
    """matmul FLOPs of ONE lean forward at this flat length, counted not guessed."""
    from torch.utils.flop_counter import FlopCounterMode
    f = FastDenoiser(model, L, device=dev, capture=False, manual_graph=False,
                     mode="default", preset=a.preset, batch_size=batch_size)
    f._static_in = f._blank_inputs()
    args = f._args(False, False)
    with torch.no_grad():
        # eager, NOT the compiled callable: FlopCounterMode is a dispatch mode
        # and inductor's fused kernels are invisible to it.
        with FlopCounterMode(display=False) as fc:
            model.diffusion.forward_core(*args)
    return float(fc.get_total_flops())


gf_b1 = flops_for(1)
gf_bB = flops_for(B)
torch.cuda.empty_cache()

group = GroupFastDenoiser(model, L, group_size=G, device=dev, preset=a.preset,
                          batch_size=B).warmup(pair_cond=False)
lean = group._graphs.get((False, False)) or group._graphs.get((False, True))
graph = lean[0]
for _ in range(10):
    graph.replay()
torch.cuda.synchronize()
s, e = torch.cuda.Event(True), torch.cuda.Event(True)
s.record()
for _ in range(a.reps):
    graph.replay()
e.record()
torch.cuda.synchronize()
ms = s.elapsed_time(e) / a.reps

per_replay = G * B
executed = G * gf_bB / (ms / 1000.0)
useful = G * B * gf_b1 / (ms / 1000.0)
peak = PEAK_TF32_MEASURED if a.preset == "fast" else PEAK_FP32
print("ROOF "
      f"preset={a.preset} L={L} B={B} G={G} designs_per_replay={per_replay} "
      f"replay_lean_ms={ms:.4f} "
      f"gflop_b1={gf_b1/1e9:.4f} gflop_bB={gf_bB/1e9:.4f} "
      f"flop_overhead_x={gf_bB/(B*gf_b1):.4f} "
      f"executed_TFLOPs={executed/1e12:.3f} useful_TFLOPs={useful/1e12:.3f} "
      f"executed_frac_fp32peak={executed/PEAK_FP32:.4f} "
      f"useful_frac_fp32peak={useful/PEAK_FP32:.4f} "
      f"executed_frac_presetpeak={executed/peak:.4f} "
      f"useful_frac_presetpeak={useful/peak:.4f} "
      f"peak_reserved_MiB={torch.cuda.max_memory_reserved(dev)/2**20:.1f}",
      flush=True)
