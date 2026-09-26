"""N016/M9 - how much of the H100 does ONE forward actually use?

M5 showed the model's individual kernels speed up 1.7-3.9x when independent
copies are replayed on forked streams, i.e. each kernel leaves most of the GPU
idle. The decisive whole-model version of that question: capture k INDEPENDENT
copies of the compiled forward into ONE cuda graph, each on its own stream, and
measure the per-forward cost.

  if per-forward cost is flat in k   -> the GPU is idle during a single forward
                                        and the 3.4 ms is latency, not capacity
  if it scales linearly              -> the forward already saturates the GPU
                                        and only fewer/bigger kernels can help

This is a hard number for the gap list: it bounds what ANY intra-forward
parallelisation (multi-stream capture, batched blocks) could ever buy, and it
is immediately usable as design-level batching.
"""
from __future__ import annotations

import argparse
import sys

import torch

ROOT = "/hnvme/workspace/v181bb10-salad/n016-floor"
sys.path.insert(0, ROOT)

from salad_torch.config import get_config                      # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.fast import FastDenoiser, PRESETS, _CORE_ARGS      # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="exact")
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--kmax", type=int, default=8)
a = ap.parse_args()

dev = torch.device("cuda")
cfg = get_config(a.config)
L = a.length
torch.set_float32_matmul_precision("high" if PRESETS[a.preset].get("tf32") else "highest")
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"),
           cfg, strict=True)

fd = FastDenoiser(m, L, device=dev, preset=a.preset)
compiled = fd._compiled


def make_args():
    st = fd._blank_inputs()
    st["pos_noised"] = torch.randn(L, 5 + cfg.augment_size, 3, device=dev) * 4
    st["prev_pos"] = torch.randn(L, 5 + cfg.augment_size, 3, device=dev) * 4
    st["pos_gt"] = st["prev_pos"].clone()
    return [st[k] for k in _CORE_ARGS] + [True, False, False]


def ev(fn, reps=100, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


base = None
with torch.no_grad():
    argsets = [make_args() for _ in range(a.kmax)]
    for _ in range(4):                       # compile once, outside any capture
        compiled(*argsets[0])
    torch.cuda.synchronize()
    for k in (1, 2, 4, a.kmax):
        if k > a.kmax:
            continue
        streams = [torch.cuda.Stream() for _ in range(k)] if k > 1 else None

        def build():
            if streams is None:
                compiled(*argsets[0])
                return
            cur = torch.cuda.current_stream()
            for s_ in streams:
                s_.wait_stream(cur)
            for i in range(k):
                with torch.cuda.stream(streams[i]):
                    compiled(*argsets[i])
            for s_ in streams:
                cur.wait_stream(s_)

        build()
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            build()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                build()
        except Exception as exc:
            print(f"M9 k={k} CAPTURE FAILED: {str(exc)[:160]}", flush=True)
            continue
        ms = ev(g.replay, 60)
        if base is None:
            base = ms
        print(f"M9 preset={a.preset} L={L} k={k:<2} total={ms:8.4f} ms  "
              f"per_forward={ms/k:7.4f} ms  speedup_vs_serial={base*k/ms:5.2f}x  "
              f"efficiency={(base/(ms/k)):5.2f}x", flush=True)
        del g
        torch.cuda.synchronize()
