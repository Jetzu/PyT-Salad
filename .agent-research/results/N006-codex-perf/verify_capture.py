"""Compiled manual CUDA graph versus the eager tensor core, fixed random fields."""
import os
import sys
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "codex"))
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser, _CORE_ARGS
from salad_torch.modules.diffusion import uniform_draw_count

torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda")
L = 100
for config_name in ("default_vp",):
    cfg = get_config(config_name)
    m = StructureDiffusionPredict(cfg).eval().to(dev)
    load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{config_name}-200k.npz"), cfg, strict=True)
    A = 14 if cfg.encode_atom14 else 5 + cfg.augment_size
    z = torch.zeros(L, dtype=torch.long, device=dev)
    pos = torch.randn(L, A, 3, device=dev) * 4
    d = dict(pos=pos, pos_noised=pos, aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
             seq=torch.full((L,), 20, dtype=torch.long, device=dev),
             seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
             residue_index=torch.arange(L, device=dev), chain_index=z, batch_index=z,
             mask=torch.ones(L, dtype=torch.bool, device=dev),
             cyclic_mask=torch.zeros(L, dtype=torch.bool, device=dev),
             t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev))
    prev = {"pos": torch.zeros(L, A, 3, device=dev),
            "local": torch.zeros(L, cfg.local_size, device=dev)}
    U = torch.rand(uniform_draw_count(cfg), L, L, device=dev).clamp_(1e-6, 1-1e-6)
    with torch.no_grad():
        cond = m.prepare_condition(d)
        pc = cond["pair_condition"]
        ref = m.diffusion.forward_core(d["pos_noised"], prev["pos"], prev["local"],
                cond["condition"], pc["dmap"], pc["omap"], pc["dmap_mask"], pc["flags"],
                d["t_pos"], d["t_seq"], d["seq_noised"], d["aa_gt"], d["pos"],
                d["residue_index"], d["chain_index"], d["batch_index"], d["mask"],
                None, U)
        f = FastDenoiser(m, L, device=dev, mode="default", manual_graph=True).warmup()
        got = f(d, prev, uniform=U)[0]
        direct = f._compiled(*(f._static_in[k] for k in _CORE_ARGS))
        torch.cuda.synchronize()
    pos_error = float((got["pos"] - ref[0]).norm(dim=-1).max())
    local_error = float((got["local"] - ref[1]).abs().max())
    aa_diff = int((got["aatype"] != ref[10]).sum())
    print(config_name, "pos_A", pos_error, "local", local_error, "aatype_diff", aa_diff, flush=True)
    graph_direct_pos = float((got["pos"] - direct[0]).norm(dim=-1).max())
    graph_direct_local = float((got["local"] - direct[1]).abs().max())
    print("graph_vs_direct_pos_A", graph_direct_pos,
          "graph_vs_direct_local", graph_direct_local, flush=True)
    del f, m
