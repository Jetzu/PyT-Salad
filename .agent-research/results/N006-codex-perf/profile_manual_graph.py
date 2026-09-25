"""Independent one-length CUDA event and kernel profile of the steady-state core."""
import os
import sys
import time
from collections import Counter
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "codex"))
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser

torch.set_float32_matmul_precision("highest")
L = 100
dev = torch.device("cuda")
cfg = get_config("default_vp")
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
A = 5 + cfg.augment_size
z = torch.zeros(L, dtype=torch.long, device=dev)
pos = torch.randn(L, A, 3, device=dev) * 4
d = dict(pos=pos, pos_noised=pos, aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
         residue_index=torch.arange(L, device=dev), chain_index=z, batch_index=z,
         mask=torch.ones(L, dtype=torch.bool, device=dev),
         cyclic_mask=torch.zeros(L, dtype=torch.bool, device=dev),
         t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev))
p = {"pos": torch.zeros(L, A, 3, device=dev),
     "local": torch.zeros(L, cfg.local_size, device=dev)}

with torch.no_grad():
    f = FastDenoiser(m, L, device=dev, mode="default", manual_graph=True).warmup()
    for _ in range(10):
        f(d, p)
    torch.cuda.synchronize()
    for reps in (50, 100):
        t0 = time.perf_counter()
        for _ in range(reps):
            f(d, p)
        torch.cuda.synchronize()
        print(f"wall_ms reps={reps} value={(time.perf_counter()-t0)*1000/reps:.4f}", flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            f(d, p)
        torch.cuda.synchronize()

events = prof.events()
kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
counts = Counter()
durations = Counter()
for e in kernels:
    n = e.name.lower()
    kind = ("gemm" if any(s in n for s in ("gemm", "sgemm", "matmul", "mm_"))
            else "memcpy" if "memcpy" in n else "triton" if "triton" in n
            else "sort" if "sort" in n else "other")
    counts[kind] += 1
    durations[kind] += e.self_device_time_total / 1000
print("kernel_count", len(kernels), "per_forward", len(kernels)/5)
print("kernel_categories", dict(counts))
print("kernel_ms_total_5_forwards", dict(durations))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25,
                                max_name_column_width=95))
