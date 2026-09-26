"""N020: whole-forward benchmark + kernel census. ONE arm, ONE length, ONE process.

Compiling several shapes or configs in one process degrades cudagraphs and once
produced a completely false "flat 17.7 ms" result (N009/N012 method note), so
the sbatch puts every (arm, length) in its own python.

Arms are selected by the SHIPPED env flags, not by a private switch:
  SALAD_OPT_MM=einsum|mm|addmm   HkLinear's contraction
  SALAD_OPT_GEMMTPL=0|1          inductor's matmul templates (max_autotune_gemm)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.profiler import ProfilerActivity, profile
from x_common import build

torch.set_grad_enabled(False)
TAG = os.environ.get("N020_TAG", "?")
L = int(os.environ.get("N020_LEN", "100"))
PRESET = os.environ.get("N020_PRESET", "exact")
REPS = int(os.environ.get("N020_REPS", "3"))

m, cfg, _ = build(L, device="cuda")
from salad_torch import opt                                       # noqa: E402
from salad_torch.modules.fast import FastDenoiser                 # noqa: E402

dev = torch.device("cuda")
A = 5 + cfg.augment_size
z = torch.zeros(L, dtype=torch.long, device=dev)
g = torch.Generator(device="cuda").manual_seed(0)
pos = torch.randn(L, A, 3, device=dev, generator=g) * 4
data = dict(pos=pos, pos_noised=pos,
            aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
            seq=torch.full((L,), 20, dtype=torch.long, device=dev),
            seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
            residue_index=torch.arange(L, device=dev), chain_index=z, batch_index=z,
            mask=torch.ones(L, dtype=torch.bool, device=dev),
            cyclic_mask=torch.zeros(L, dtype=torch.bool, device=dev),
            t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev))
prev = {"pos": torch.zeros(L, A, 3, device=dev),
        "local": torch.zeros(L, cfg.local_size, device=dev)}


def bench(fn, it=60, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e3


out = {"tag": TAG, "L": L, "preset": PRESET, "mm_mode": opt.MM_MODE,
       "gemm_templates": opt.GEMM_TEMPLATES,
       "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
t0 = time.perf_counter()
fast = FastDenoiser(m, L, device=dev, preset=PRESET).warmup(pair_cond=False)
out["compile_s"] = time.perf_counter() - t0
out["full_ms"] = [bench(lambda: fast(data, prev)) for _ in range(REPS)]
out["lean_ms"] = [bench(lambda: fast(data, prev, need_aa=False)) for _ in range(REPS)]

graph, _ = fast._graphs[(False, False)]
for _ in range(5):
    graph.replay()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
ev = [e for e in p.key_averages()
      if e.device_type.name == "CUDA" and e.device_time_total > 0]
census = {k: [0, 0.0] for k in ("triton_tem", "triton", "gemm", "memcpy", "other")}
for e in ev:
    low = e.key.lower()
    k = ("triton_tem" if low.startswith("triton_tem") else
         "triton" if low.startswith("triton") else
         "memcpy" if "memcpy" in low or "memset" in low else
         "gemm" if ("gemm" in low or "cutlass" in low or "cublas" in low
                    or "gemv" in low or "splitk" in low) else "other")
    census[k][0] += e.count / 10
    census[k][1] += e.device_time_total / 10
out["census"] = census
out["kernels"] = sum(v[0] for v in census.values())
out["gpu_busy_us"] = sum(v[1] for v in census.values())
out["top20"] = [(e.key[:56], round(e.count / 10, 1), round(e.device_time_total / 10, 1))
                for e in sorted(ev, key=lambda e: -e.device_time_total)[:20]]
a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
a.record()
for _ in range(200):
    graph.replay()
b.record()
b.synchronize()
out["replay_us"] = a.elapsed_time(b) * 1000 / 200
r, _ = fast(data, prev, need_aa=False)
out["pos_sum"] = float(r["pos"].double().sum())
out["pos_absmax"] = float(r["pos"].abs().max())
print("RESULT " + json.dumps(out), flush=True)
