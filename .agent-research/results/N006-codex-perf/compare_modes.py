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
from salad_torch.modules.fast import FastDenoiser
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
        original = FastDenoiser(m, L, device=dev, mode="reduce-overhead").warmup()
        old = original(d, prev, uniform=U)[0]
        torch.cuda.synchronize()
    pos_error = float((got["pos"] - ref[0]).norm(dim=-1).max())
    local_error = float((got["local"] - ref[1]).abs().max())
    aa_diff = int((got["aatype"] != ref[10]).sum())
    old_pos_error = float((old["pos"] - ref[0]).norm(dim=-1).max())
    old_local_error = float((old["local"] - ref[1]).abs().max())
    graph_vs_old_pos = float((got["pos"] - old["pos"]).norm(dim=-1).max())
    graph_vs_old_local = float((got["local"] - old["local"]).abs().max())
    print(config_name, "manual_vs_eager_pos_A", pos_error,
          "old_vs_eager_pos_A", old_pos_error,
          "manual_vs_old_pos_A", graph_vs_old_pos,
          "manual_vs_eager_local", local_error,
          "old_vs_eager_local", old_local_error,
          "manual_vs_old_local", graph_vs_old_local,
          "aatype_diff", aa_diff, flush=True)
    del f, m
