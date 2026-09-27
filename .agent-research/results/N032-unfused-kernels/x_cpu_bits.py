"""N032 control: are `benchmark_fusion`/`realize_reads_threshold` bit-exact?

CPU is deterministic by construction (no autotune race, no TF32), so a bitwise
comparison of the compiled forward with and without the flag is a real answer.
Two processes, one compile each, dump tensors, compare offline.

  N032_BENCHFUSION=1        torch._inductor.config.benchmark_fusion = True
  N032_REALIZEREADS=<int>   torch._inductor.config.realize_reads_threshold
"""
import os
import sys

import torch

TREE = os.environ["SALAD_TREE"]
sys.path.insert(0, TREE)
OUT = os.environ["N032_DUMP"]

import torch._inductor.config as ind                               # noqa: E402

if os.environ.get("N032_BENCHFUSION") == "1":
    ind.benchmark_fusion = True
if os.environ.get("N032_REALIZEREADS"):
    ind.realize_reads_threshold = int(os.environ["N032_REALIZEREADS"])

from salad_torch.config import get_config                          # noqa: E402
from salad_torch.convert.haiku_to_torch import (load_model,        # noqa: E402
                                                load_param_npz)
from salad_torch.modules.model import StructureDiffusionPredict    # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(0)
CFG = "default_vp"
L = int(os.environ.get("N032_LEN", "64"))
cfg = get_config(CFG)
m = StructureDiffusionPredict(cfg).eval()
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{CFG}-200k.npz"),
           cfg, strict=True)
A = 5 + cfg.augment_size
g = torch.Generator().manual_seed(7)
pos = torch.randn(L, A, 3, generator=g) * 4
z = torch.zeros(L, dtype=torch.long)
core = torch.compile(m.diffusion.forward_core, dynamic=False)
from salad_torch.modules.diffusion import uniform_draw_count       # noqa: E402
nd = uniform_draw_count(cfg)
args = dict(
    pos_noised=pos, prev_pos=torch.zeros(L, A, 3), prev_local=torch.zeros(L, cfg.local_size),
    condition=torch.zeros(L, cfg.local_size), dmap=torch.zeros(L, L),
    omap=torch.zeros(L, L, 9), dmap_mask=torch.zeros(L, L, dtype=torch.bool),
    flags=torch.zeros(L, L, 3), t_pos=torch.full((L,), 0.5), t_seq=torch.ones(L),
    seq_noised=torch.full((L,), 20, dtype=torch.long),
    aa_gt=torch.full((L,), 20, dtype=torch.long), pos_gt=pos,
    resi=torch.arange(L), chain=z, batch=z, mask=torch.ones(L, dtype=torch.bool),
    cyclic_mask=None,
    uniform=torch.rand(nd, L, L, generator=g).clamp_(1e-6, 1 - 1e-6))
order = ("pos_noised", "prev_pos", "prev_local", "condition", "dmap", "omap",
         "dmap_mask", "flags", "t_pos", "t_seq", "seq_noised", "aa_gt", "pos_gt",
         "resi", "chain", "batch", "mask", "cyclic_mask", "uniform")
out = core(*[args[k] for k in order], True, True, False)
torch.save([o.detach().clone() if torch.is_tensor(o) else o for o in out], OUT)
print("wrote", OUT, "tensors", len(out),
      "benchmark_fusion=", ind.benchmark_fusion,
      "realize_reads_threshold=", ind.realize_reads_threshold, flush=True)
