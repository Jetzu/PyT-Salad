"""N017: one arm, one process, one length.

Arms are selected by environment so a single sbatch can put each in its own
process (compiling several shapes/configs in one process degrades cudagraphs -
N009/N012 method note).

  N017_CSE=1      install `fx_graph_cse` as inductor's joint_custom_post_pass
  N017_FREEZE=1   torch._inductor.config.freezing (which also CSEs, and
                  constant-folds the parameters)
  N017_IDXMEAN=0  restore index_mean's (N, F) ones_like denominator
"""
import json
import os
import sys
import time

import torch

TREE = os.environ["SALAD_TREE"]
sys.path.insert(0, TREE)
TAG = os.environ.get("N017_TAG", "?")
L = int(os.environ.get("N017_LEN", "100"))
CFG = os.environ.get("N017_CFG", "default_vp")

import torch._inductor.config as ind                              # noqa: E402

if os.environ.get("N017_CSE") == "1":
    from torch._functorch.compile_utils import fx_graph_cse

    def _cse(graph):
        gm = graph.owning_module
        gm.graph = fx_graph_cse(graph)

    ind.joint_custom_post_pass = _cse
if os.environ.get("N017_FREEZE") == "1":
    ind.freezing = True

from salad_torch.config import get_config                          # noqa: E402
from salad_torch.convert.haiku_to_torch import (load_model,        # noqa: E402
                                                load_param_npz)
from salad_torch.modules.fast import FastDenoiser                  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict    # noqa: E402

dev = torch.device("cuda")
cfg = get_config(CFG)
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{CFG}-200k.npz"),
           cfg, strict=True)
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


out = {"tag": TAG, "L": L, "cfg": CFG}
with torch.no_grad():
    t0 = time.perf_counter()
    fast = FastDenoiser(m, L, device=dev, preset="exact").warmup(pair_cond=False)
    out["compile_s"] = time.perf_counter() - t0
    out["full_ms"] = bench(lambda: fast(data, prev))
    out["lean_ms"] = bench(lambda: fast(data, prev, need_aa=False))
    # kernel census of the lean captured replay
    from torch.profiler import ProfilerActivity, profile
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
    census = {"triton": [0, 0.0], "gemm": [0, 0.0], "memcpy": [0, 0.0], "other": [0, 0.0]}
    for e in ev:
        low = e.key.lower()
        k = ("triton" if low.startswith("triton") else
             "memcpy" if "memcpy" in low or "memset" in low else
             "gemm" if ("gemm" in low or "cutlass" in low or "cublas" in low
                        or "gemv" in low or "splitk" in low) else "other")
        census[k][0] += e.count / 10
        census[k][1] += e.device_time_total / 10
    out["census"] = census
    out["kernels"] = sum(v[0] for v in census.values())
    out["gpu_busy_us"] = sum(v[1] for v in census.values())
    a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    a.record()
    for _ in range(200):
        graph.replay()
    b.record(); b.synchronize()
    out["replay_us"] = a.elapsed_time(b) * 1000 / 200
    # a cheap output fingerprint so an arm that changes the numbers is visible
    r, _ = fast(data, prev, need_aa=False)
    out["pos_sum"] = float(r["pos"].double().sum())
    out["pos_absmax"] = float(r["pos"].abs().max())
print("RESULT " + json.dumps(out), flush=True)
