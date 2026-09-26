"""N017 step 1: profile the lean captured step and dump inductor's debug artefacts.

One length (100) per process - compiling several shapes in one process degrades
cudagraphs (N009/N012 method note).
"""
import json
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

TREE = os.environ.get("SALAD_TREE", os.getcwd())
sys.path.insert(0, TREE)
OUT = os.environ["N017_OUT"]
os.makedirs(OUT, exist_ok=True)

from salad_torch.config import get_config                       # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.fast import FastDenoiser               # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict  # noqa: E402

L = int(os.environ.get("N017_LEN", "100"))
cfg = get_config("default_vp")
m = StructureDiffusionPredict(cfg).eval().cuda()
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"),
           cfg, strict=True)
f = FastDenoiser(m, L, device="cuda", manual_graph=True, mode="default", preset="exact")
f._static_in = f._blank_inputs()

results = {}
with torch.no_grad():
    f._build(False, False, iters=3)   # the LEAN step, no pair conditioning
    g, st = f._graphs[(False, False)]
    args = f._args(False, False)

    for label, fn in (("captured", g.replay), ("noncaptured", lambda: f._compiled(*args))):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
        ev = [e for e in p.key_averages()
              if e.device_type.name == "CUDA" and e.device_time_total > 0]
        rows = sorted(({"name": e.key, "count": e.count / 10,
                        "us_total": e.device_time_total / 10,
                        "us_each": e.device_time_total / max(e.count, 1)}
                       for e in ev), key=lambda r: -r["us_total"])
        results[label] = rows
        print(label, "kernels", sum(r["count"] for r in rows),
              "gpu_busy_us", round(sum(r["us_total"] for r in rows), 1), flush=True)

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    for _ in range(200):
        g.replay()
    ev_end.record()
    ev_end.synchronize()
    results["replay_us"] = ev_start.elapsed_time(ev_end) * 1000 / 200
    print("REPLAY_EVENT_US", results["replay_us"], flush=True)

with open(os.path.join(OUT, "profile_lean.json"), "w") as fh:
    json.dump(results, fh, indent=1)
print("wrote", os.path.join(OUT, "profile_lean.json"))

# ---------------------------------------------------------------- floor probe
# What does ONE extra kernel cost inside a captured graph? The audit's whole
# thesis is that the triton kernels are at the launch floor, so the marginal
# value of removing one has to be measured, not assumed.
import triton                                                     # noqa: E402
import triton.language as tl                                      # noqa: E402


@triton.jit
def _noop(p, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(p + off, tl.load(p + off, mask=m) + 1.0, mask=m)


floor = {}
buf = torch.zeros(256, device="cuda")
for nk in (0, 50, 100, 200, 400):
    gph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(nk):
                _noop[(1,)](buf, 256, BLOCK=256)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(gph):
        for _ in range(nk):
            _noop[(1,)](buf, 256, BLOCK=256)
    for _ in range(5):
        gph.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(100):
        gph.replay()
    b.record(); b.synchronize()
    floor[nk] = a.elapsed_time(b) * 1000 / 100
    print("FLOOR", nk, round(floor[nk], 2), "us", flush=True)
results["floor_us"] = floor
with open(os.path.join(OUT, "profile_lean.json"), "w") as fh:
    json.dump(results, fh, indent=1)
