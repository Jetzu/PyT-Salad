"""Mirror of jax_dump.py on the torch side: dump every DiffusionBlock intermediate."""
from __future__ import annotations
import os, sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)

from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules import blocks as B
from salad_torch.diffusion.precond import preconditioning_scale_factors

torch.set_grad_enabled(False)
REC = {}


def rec(name, x):
    REC.setdefault(name, []).append(np.asarray(x.detach().cpu().numpy()))
    return x


def rec_forward(self, features, pos, prev_pos, condition, pair_condition, time,
                resi, chain, batch, mask, current_neighbours, cond_neighbours,
                cyclic_mask=None, init_pos=None):
    c = self.config
    sigma = c.sigma_data
    rec("block_in_features", features); rec("block_in_pos", pos)
    rec("block_condition", condition); rec("block_prev_pos", prev_pos)
    rec("current_neighbours", current_neighbours); rec("cond_neighbours", cond_neighbours)

    pair, pair_mask = self.pair_1(pos, pair_condition, current_neighbours,
                                  resi, chain, batch, mask,
                                  cyclic_mask=cyclic_mask, init_pos=init_pos)
    rec("pair_1", pair)
    ln1 = self.layer_norm_1(features); rec("ln_1", ln1)
    a1 = self.attn_1(ln1, pos / sigma, pair, pair_mask, current_neighbours)
    rec("attn_1", a1)
    features = features + a1

    pair, pair_mask = self.pair_2(prev_pos, pair_condition, cond_neighbours,
                                  resi, chain, batch, mask,
                                  cyclic_mask=cyclic_mask, init_pos=init_pos)
    rec("pair_2", pair)
    ln3 = self.layer_norm_3(features); rec("ln_3", ln3)
    a2 = self.attn_2(ln3, prev_pos / sigma, pair, pair_mask, cond_neighbours)
    rec("attn_2", a2)
    features = features + a2

    ln4 = self.layer_norm_4(features); rec("ln_4", ln4)
    u = self.update(ln4, pos, chain, batch, mask, condition)
    rec("update_out", u)
    features = features + u

    local_norm = self.layer_norm_5(features); rec("local_norm", local_norm)
    out_scale = preconditioning_scale_factors(c, time[:, None], sigma)["out"]
    pos = self.update_positions(pos, local_norm, out_scale)
    rec("block_out_pos", pos); rec("block_out_features", features)
    return features, pos.to(local_norm.dtype)


def _t(a, dtype):
    if a.dtype == np.bool_:
        return torch.from_numpy(a)
    if np.issubdtype(a.dtype, np.integer):
        return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a.astype(np.float64 if dtype == torch.float64 else np.float32)).to(dtype)


def main(cfg_name, fix_path, out_path, dtype=torch.float32, pos_delta=None):
    B.DiffusionBlock.forward = rec_forward
    cfg = get_config(cfg_name)
    D = np.load(fix_path)
    model = StructureDiffusionPredict(cfg).eval()
    load_model(model, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{cfg_name}-200k.npz"),
               cfg, strict=True)
    model.to(dtype)
    data = {k[3:]: _t(D[k], dtype) for k in D.files
            if k.startswith("in_") and k not in ("in_prev_pos", "in_prev_local")}
    for k in ("pos_noised", "seq_noised", "t_pos", "t_seq"):
        data[k] = _t(D[f"noised_{k}"], dtype)
    if pos_delta is not None:
        data["pos_noised"] = data["pos_noised"] + torch.from_numpy(pos_delta).to(dtype)
    prev = {"pos": _t(D["in_prev_pos"], dtype), "local": _t(D["in_prev_local"], dtype)}
    uniform = _t(D["fixed_uniform"], dtype)
    result, _ = model(data, prev, predict_aa=True, uniform=uniform)
    out = {f"{k}#{i}": v for k, vs in REC.items() for i, v in enumerate(vs)}
    out["final_pos"] = result["pos"].numpy()
    out["final_local"] = result["local"].numpy()
    np.savez(out_path, **out)
    print("wrote", out_path)


if __name__ == "__main__":
    dt = torch.float64 if (len(sys.argv) > 4 and sys.argv[4] == "f64") else torch.float32
    main(sys.argv[1], sys.argv[2], sys.argv[3], dt)
