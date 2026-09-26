"""N021 - memory traffic of ONE lean forward at a given batch_size.

Uncaptured (capture=False, manual_graph=False) so Nsight Compute can see the
individual kernels, exactly as N016's x_ncu.py did - the numbers are therefore
directly comparable with N016's 890.94 MiB DRAM / 4814.85 MiB L2 baseline.
"""
import os
import sys

import torch

ROOT = os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n021tree")
sys.path.insert(0, ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.diffusion.sampling import make_initial_data, pack_designs
from salad_torch.diffusion.schedules import parse_num_aa
from salad_torch.modules.fast import FastDenoiser
from salad_torch.modules.model import StructureDiffusionPredict

L = int(os.environ.get("SALAD_L", "100"))
B = int(os.environ.get("SALAD_B", "1"))
PRESET = os.environ.get("SALAD_PRESET", "fast")
dev = torch.device("cuda")
cfg = get_config("default_vp")
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"),
           cfg, strict=True)
num_aa, resi, chain, _, _ = parse_num_aa(str(L))
per = [make_initial_data(num_aa, resi, chain, None, device=dev) for _ in range(B)]
data, prev, _lens = pack_designs([d for d, _ in per], [p for _, p in per])
data["pos"] = torch.randn_like(data["pos"]) * 4
data["pos_noised"] = data["pos"]
data["t_pos"] = torch.full_like(data["t_pos"], 0.5)
with torch.no_grad():
    f = FastDenoiser(m, L, device=dev, capture=False, manual_graph=False,
                     mode="default", preset=PRESET, batch_size=B)
    f.warmup(pair_cond=False)
    f.pin_condition(data)
    for _ in range(5):
        f(data, prev, need_aa=False)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("measure")
    f(data, prev, need_aa=False)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print(f"NCU-SCRIPT-DONE L={L} B={B}", flush=True)
