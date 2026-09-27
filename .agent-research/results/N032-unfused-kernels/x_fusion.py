"""N028: dump everything needed to answer WHY 331 Triton kernels carry one op.

One arm, one process, one length (the N009/N012 method note). Writes:
  <OUT>/output_code.py, ir_pre_fusion.txt, ir_post_fusion.txt  (inductor debug)
  <OUT>/fusion.log                                             (TORCH_LOGS=fusion)
  <OUT>/census.json    kernel name -> (launches, us) from the CAPTURED replay
"""
import json
import os
import shutil
import sys
import time

TREE = os.environ["SALAD_TREE"]
sys.path.insert(0, TREE)
OUT = os.environ["N032_OUT"]
L = int(os.environ.get("N032_LEN", "100"))
CFG = os.environ.get("N032_CFG", "default_vp")
PRESET = os.environ.get("N032_PRESET", "exact")
os.makedirs(OUT, exist_ok=True)

import torch                                                   # noqa: E402
import torch._inductor.config as ind                           # noqa: E402

ind.trace.enabled = True
ind.trace.debug_dir = os.path.join(OUT, "tcdebug")

import logging                                                 # noqa: E402
# `set_logs(fusion=...)` takes a BOOL for an artifact, not a level - passing
# logging.DEBUG raises. Enabling the artifact is what sets it to DEBUG.
torch._logging.set_logs(fusion=True)
_fh = logging.FileHandler(os.path.join(OUT, "fusion.log"), mode="w")
_fh.setLevel(logging.DEBUG)
_lg = logging.getLogger("torch._inductor.scheduler")
_lg.addHandler(_fh)
_lg.setLevel(logging.DEBUG)

from salad_torch.config import get_config                       # noqa: E402
from salad_torch.convert.haiku_to_torch import (load_model,     # noqa: E402
                                                load_param_npz)
from salad_torch.modules.fast import FastDenoiser               # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict  # noqa: E402

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

with torch.no_grad():
    t0 = time.perf_counter()
    fast = FastDenoiser(m, L, device=dev, preset=PRESET).warmup(pair_cond=False)
    compile_s = time.perf_counter() - t0
    graph, _ = fast._graphs[(False, False)]
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
    rows = []
    for e in p.key_averages():
        if e.device_type.name == "CUDA" and e.device_time_total > 0:
            rows.append(dict(name=e.key, launches=e.count / 10,
                             us=e.device_time_total / 10))

json.dump(dict(L=L, cfg=CFG, preset=PRESET, compile_s=compile_s, rows=rows),
          open(os.path.join(OUT, "census.json"), "w"), indent=1)

# hoist the inductor artefacts out of the nested debug dir
for root, _dirs, files in os.walk(os.path.join(OUT, "tcdebug")):
    for f in files:
        if f in ("output_code.py", "ir_pre_fusion.txt", "ir_post_fusion.txt",
                 "fx_graph_readable.py"):
            dst = os.path.join(OUT, f)
            if not os.path.exists(dst):
                shutil.copy(os.path.join(root, f), dst)
print("DUMPED", OUT, "compile_s=%.1f" % compile_s,
      "kernels=%.0f" % sum(r["launches"] for r in rows), flush=True)
