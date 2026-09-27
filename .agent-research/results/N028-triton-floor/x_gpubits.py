"""N028: GPU bitwise control under SALAD_OPT_DETSEG=1.

`aggressive_fusion` was admitted to `exact` on exactly this pair of controls
(N013): bit-identical on CPU (deterministic by construction) AND bit-identical
on the H100 under the deterministic segment reduction, which removes the only
run-to-run noise source. A CPU-only bitwise result does not prove the GPU
kernels inductor emits for the two arms compute the same bits.
Dumps one tensor file per arm; compare with x_cmp_bits.py.
"""
import os
import sys

import torch

TREE = os.environ["SALAD_TREE"]
sys.path.insert(0, TREE)
OUT = os.environ["N028_DUMP"]
L = int(os.environ.get("N028_LEN", "100"))

import torch._inductor.config as ind                              # noqa: E402
from torch._functorch.compile_utils import fx_graph_cse           # noqa: E402


def _cse(graph):
    graph.owning_module.graph = fx_graph_cse(graph)


ind.joint_custom_post_pass = _cse
ind.aggressive_fusion = True

import salad_torch.opt as opt                                     # noqa: E402
opt.DET_SEGMENT = True          # the whole point: no atomics, no run-to-run noise
from salad_torch.config import get_config                         # noqa: E402
from salad_torch.convert.haiku_to_torch import (load_model,       # noqa: E402
                                                load_param_npz)
from salad_torch.modules.diffusion import uniform_draw_count      # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict   # noqa: E402

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda")
cfg = get_config("default_vp")
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"),
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
core = torch.compile(m.diffusion.forward_core, dynamic=False)
out = core(*[args[k] for k in order], True, True, False)
torch.save([o.detach().clone().cpu() if torch.is_tensor(o) else o for o in out], OUT)
print("wrote", OUT, "IPA_MULSUM", opt.IPA_MULSUM, "PAIR_FEAT_MERGE",
      opt.PAIR_FEAT_MERGE, "FRAME_MULSUM", getattr(opt, "FRAME_MULSUM", None),
      flush=True)
