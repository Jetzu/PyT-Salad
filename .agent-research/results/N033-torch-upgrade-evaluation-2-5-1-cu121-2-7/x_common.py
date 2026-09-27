"""N020 shared: build the real `forward_core` argument list for a length L.

Copied from N017's x_cpu_bits.py so the two items' controls are comparable.
"""
import os
import sys

import torch

TREE = os.environ["SALAD_TREE"]
if TREE not in sys.path:
    sys.path.insert(0, TREE)

ORDER = ("pos_noised", "prev_pos", "prev_local", "condition", "dmap", "omap",
         "dmap_mask", "flags", "t_pos", "t_seq", "seq_noised", "aa_gt", "pos_gt",
         "resi", "chain", "batch", "mask", "cyclic_mask", "uniform")
NAMES = ("pos", "local", "trajectory", "aa", "aa_features", "dssp",
         "distogram_neighbours", "raw_angles", "angles", "angle_pos", "aatype")


def build(L, cfg_name="default_vp", device="cpu"):
    from salad_torch.config import get_config
    from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
    from salad_torch.modules.diffusion import uniform_draw_count
    from salad_torch.modules.model import StructureDiffusionPredict

    dev = torch.device(device)
    cfg = get_config(cfg_name)
    m = StructureDiffusionPredict(cfg).eval().to(dev)
    load_model(m, load_param_npz(
        f"/hnvme/workspace/v181bb10-salad/{cfg_name}-200k.npz"), cfg, strict=True)
    A = 5 + cfg.augment_size
    g = torch.Generator(device=device).manual_seed(7)
    pos = torch.randn(L, A, 3, generator=g, device=dev) * 4
    z = torch.zeros(L, dtype=torch.long, device=dev)
    nd = uniform_draw_count(cfg)
    args = dict(
        pos_noised=pos, prev_pos=torch.zeros(L, A, 3, device=dev),
        prev_local=torch.zeros(L, cfg.local_size, device=dev),
        condition=torch.zeros(L, cfg.local_size, device=dev),
        dmap=torch.zeros(L, L, device=dev), omap=torch.zeros(L, L, 9, device=dev),
        dmap_mask=torch.zeros(L, L, dtype=torch.bool, device=dev),
        flags=torch.zeros(L, L, 3, device=dev),
        t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev),
        seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
        aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev), pos_gt=pos,
        resi=torch.arange(L, device=dev), chain=z, batch=z,
        mask=torch.ones(L, dtype=torch.bool, device=dev), cyclic_mask=None,
        uniform=torch.rand(nd, L, L, generator=g, device=dev).clamp_(1e-6, 1 - 1e-6))
    return m, cfg, [args[k] for k in ORDER]
