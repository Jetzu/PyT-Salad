import sys, pickle
import numpy as np, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
torch.set_grad_enabled(False)

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

cfg = dotdict(local_size=128, pair_size=64, resi_dual=False, factor=4, heads=8,
              key_size=32, multi_query=False, diffusion_depth=6, sigma_data=10.0,
              preconditioning="vp", augment_size=15)

import torch.nn.functional as _F
import salad_torch.modules.basic as _basic, salad_torch.modules.blocks as _blocks
_tanh_gelu = lambda x: _F.gelu(x, approximate="tanh")
_basic.gelu = _tanh_gelu; _blocks.gelu = _tanh_gelu
from salad_torch.modules.blocks import DiffusionBlock
blk = DiffusionBlock(cfg, local_size=128, atom_count=20, use_init_pos=False).eval()
from salad_torch.modules.basic import MLP as _MLP
for _m in blk.modules():
    if isinstance(_m, _MLP) and _m.activation is not _F.relu:
        _m.activation = _tanh_gelu

D = np.load("/tmp/claude-215432/-home-hpc-v181bb-v181bb10/0eeb4a3c-fee2-4b4d-8ef9-9555175d051e/scratchpad/block_ref.npz")
NPZ = np.load("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
P = "structure_diffusion/diffusion/diffusion_stack/__layer_stack_with_state/diffusion_block/"
L = 0
def hk(scope):
    out = {}
    for nm in ("w","b","scale","offset","gamma"):
        key = P + scope + "/" + nm
        if key in NPZ:
            out[nm] = NPZ[key][L]
    return out

MAP = {}
def lin(port, scope): MAP[port] = ("lin", scope)
def ln(port, scope):  MAP[port] = ("ln", scope)

for i, name in enumerate(["relpos","dmap","flags","dist","dir","rot","pairvec"]):
    lin(f"pair_1.linear_{name}", "linear" if i==0 else f"linear_{i}")
    lin(f"pair_2.linear_{name}", f"linear_{7+i}")
ln("pair_1.layer_norm", "layer_norm"); ln("pair_2.layer_norm", "layer_norm_2")
lin("pair_1.mlp.layers.0", "mlp/linear"); lin("pair_1.mlp.layers.1", "mlp/linear_1")
lin("pair_2.mlp.layers.0", "mlp_1/linear"); lin("pair_2.mlp.layers.1", "mlp_1/linear_1")
ln("layer_norm_1","layer_norm_1"); ln("layer_norm_3","layer_norm_3")
ln("layer_norm_4","layer_norm_4"); ln("layer_norm_5","layer_norm_5")
for pa, ha in (("attn_1","sparse_structure_attn"), ("attn_2","sparse_structure_attn_1")):
    b = f"{pa}.ada_point_attention."; h = f"{ha}/ada_point_attention/"
    ln(b+"layer_norm", h+"layer_norm"); ln(b+"layer_norm_1", h+"layer_norm_1")
    ln(b+"layer_norm_2", h+"layer_norm_2")
    lin(b+"qkv", h+"qkv"); lin(b+"bias", h+"bias")
    lin(b+"qkv_global.linear", h+"qkv_global/linear")
    lin(b+"project_out", h+"project_out")
    MAP[b+"gamma"] = ("gamma", h.rstrip("/"))
lin("update.linear_condition","light_global_update/linear")
lin("update.mlp.layers.0","light_global_update/mlp/linear")
lin("update.mlp.layers.1","light_global_update/mlp/linear_1")
lin("update.linear_update","light_global_update/linear_1")
lin("update.linear_gate","light_global_update/linear_2")
lin("update.linear_chain_gate","light_global_update/linear_3")
lin("update.linear_batch_gate","light_global_update/linear_4")
lin("update.linear_out","light_global_update/linear_5")
lin("update_positions.linear","linear_14")

sd = {}
for port, (kind, scope) in MAP.items():
    w = hk(scope)
    if kind == "lin":
        sd[port + ".w"] = torch.from_numpy(w["w"].copy())
        sd[port + ".b"] = torch.from_numpy(w["b"].copy())
    elif kind == "ln":
        sd[port + ".scale"] = torch.from_numpy(w["scale"].copy())
        sd[port + ".offset"] = torch.from_numpy(w["offset"].copy())
    else:
        sd[port] = torch.from_numpy(NPZ[P+scope+"/gamma"][L].copy())
missing, unexpected = blk.load_state_dict(sd, strict=False)
print("MISSING (port params with no checkpoint source):", missing)
print("UNEXPECTED:", unexpected)

def T(k, dt=None):
    a = D[k]
    if a.dtype == np.bool_: return torch.from_numpy(a)
    if np.issubdtype(a.dtype, np.integer): return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a.astype(np.float32))

pair_condition = dict(dmap=T("in_dmap"), dmap_mask=T("in_dmap_mask"), flags=T("in_flags"))
out_f, out_p = blk(
    T("in_features"), T("in_pos"), T("in_prev_pos"), T("in_condition"),
    pair_condition, T("in_time"), T("in_resi"), T("in_chain"), T("in_batch"),
    T("in_mask"), T("current_neighbours"), T("cond_neighbours"))

# also check pair features branch 1 in isolation
pair1, pm1 = blk.pair_1(T("in_pos"), pair_condition, T("current_neighbours"),
                        T("in_resi"), T("in_chain"), T("in_batch"), T("in_mask"))

def rep(name, got, ref):
    g = got.numpy() if torch.is_tensor(got) else got
    r = np.asarray(ref)
    if g.shape != r.shape:
        print(f"[FAIL] {name}: shape {g.shape} != {r.shape}"); return
    if r.dtype == np.bool_:
        nd = int((g != r).sum()); print(f"[{'PASS' if nd==0 else 'FAIL'}] {name}: {nd} differ"); return
    e = float(np.nanmax(np.abs(g - r)))
    rel = e / max(float(np.nanmax(np.abs(r))), 1e-12)
    print(f"[{'PASS' if rel < 2e-4 else 'FAIL'}] {name}: max|d|={e:.3e} rel={rel:.3e}  refscale={np.abs(r).max():.3e}")

rep("pair_features_1", pair1, D["pair1"])
rep("pair_mask_1", pm1, D["pair_mask1"])
rep("block.out_features", out_f, D["out_features"])
rep("block.out_pos", out_p, D["out_pos"])
