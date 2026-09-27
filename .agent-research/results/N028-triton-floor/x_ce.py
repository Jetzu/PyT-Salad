"""N028: COMPILED vs EAGER, whole `forward_core`, on the GPU.

K021: all four parity tests drive the EAGER path, so none of them can see a
`torch.compile` miscompile - which is exactly what P7 was (N025: 9.586 A on one
forward against a 2.0e-05 A floor). Any flag that changes what inductor fuses
needs this check, and it needs its own floor measured in the SAME job with the
flag OFF.

Arm selected by the SALAD_OPT_* env vars; prints the max|d| per output.
"""
import json
import os
import sys

import torch

TREE = os.environ["SALAD_TREE"]
sys.path.insert(0, TREE)
TAG = os.environ.get("N028_TAG", "?")
L = int(os.environ.get("N028_LEN", "100"))

import torch._inductor.config as ind                              # noqa: E402
from torch._functorch.compile_utils import fx_graph_cse           # noqa: E402


def _cse(graph):
    graph.owning_module.graph = fx_graph_cse(graph)


ind.joint_custom_post_pass = _cse
ind.aggressive_fusion = True          # what `exact`/`fast` actually ship

import salad_torch.opt as opt                                     # noqa: E402
from salad_torch.config import get_config                         # noqa: E402
from salad_torch.convert.haiku_to_torch import (load_model,       # noqa: E402
                                                load_param_npz)
from salad_torch.modules.diffusion import uniform_draw_count      # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict   # noqa: E402

torch.set_grad_enabled(False)
dev = torch.device("cuda")
CFG = "default_vp"
cfg = get_config(CFG)
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{CFG}-200k.npz"),
           cfg, strict=True)
A = 5 + cfg.augment_size
g = torch.Generator(device="cuda").manual_seed(7)
pos = torch.randn(L, A, 3, device=dev, generator=g) * 4
z = torch.zeros(L, dtype=torch.long, device=dev)
nd = uniform_draw_count(cfg)
args = dict(
    pos_noised=pos, prev_pos=torch.randn(L, A, 3, device=dev, generator=g) * 4,
    prev_local=torch.randn(L, cfg.local_size, device=dev, generator=g),
    condition=torch.zeros(L, cfg.local_size, device=dev),
    dmap=torch.zeros(L, L, device=dev), omap=torch.zeros(L, L, 9, device=dev),
    dmap_mask=torch.zeros(L, L, dtype=torch.bool, device=dev),
    flags=torch.zeros(L, L, 3, device=dev),
    t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev),
    seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
    aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev), pos_gt=pos,
    resi=torch.arange(L, device=dev), chain=z, batch=z,
    mask=torch.ones(L, dtype=torch.bool, device=dev), cyclic_mask=None,
    uniform=torch.rand(nd, L, L, device=dev, generator=g).clamp_(1e-6, 1 - 1e-6))
order = ("pos_noised", "prev_pos", "prev_local", "condition", "dmap", "omap",
         "dmap_mask", "flags", "t_pos", "t_seq", "seq_noised", "aa_gt", "pos_gt",
         "resi", "chain", "batch", "mask", "cyclic_mask", "uniform")
a = [args[k] for k in order]

eager = m.diffusion.forward_core(*a, True, True, False)
comp = torch.compile(m.diffusion.forward_core, dynamic=False)(*a, True, True, False)
NAMES = ("pos", "local", "trajectory", "aa", "aa_features", "dssp",
         "distogram_neighbours", "raw_angles", "angles", "angle_pos", "aatype")
res = {"tag": TAG, "L": L,
       "flags": {"IPA_MULSUM": opt.IPA_MULSUM,
                 "PAIR_FEAT_MERGE": opt.PAIR_FEAT_MERGE,
                 "FRAME_MULSUM": getattr(opt, "FRAME_MULSUM", None)}}
for i, (x, y) in enumerate(zip(eager, comp)):
    if not torch.is_tensor(x):
        continue
    nm = NAMES[i] if i < len(NAMES) else f"out{i}"
    d = ((x.double() - y.double()).abs().max().item() if x.is_floating_point()
         else float((x != y).sum()))
    res[nm] = d
print("CEVE " + json.dumps(res), flush=True)
