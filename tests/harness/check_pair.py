import sys
import numpy as np, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
torch.set_grad_enabled(False)
SC="/tmp/claude-215432/-home-hpc-v181bb-v181bb10/0eeb4a3c-fee2-4b4d-8ef9-9555175d051e/scratchpad/"
D = np.load(SC+"block_ref.npz"); R = np.load(SC+"pair_ref.npz")
NPZ = np.load("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
P = "structure_diffusion/diffusion/diffusion_stack/__layer_stack_with_state/diffusion_block/"
L=0
from salad_torch.geometry import features as GF
from salad_torch.modules.basic import HkLayerNorm, MLP, HkLinear
import torch.nn.functional as _F
gelu = lambda x: _F.gelu(x, approximate="tanh")

def T(k, src=D):
    a = src[k]
    if a.dtype==np.bool_: return torch.from_numpy(a)
    if np.issubdtype(a.dtype, np.integer): return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a.astype(np.float32))

def rep(name, g, r):
    g = g.numpy() if torch.is_tensor(g) else np.asarray(g)
    r = np.asarray(r)
    if g.shape != r.shape: print(f"[FAIL] {name} shape {g.shape} != {r.shape}"); return
    e = float(np.abs(g-r).max()); sc = max(float(np.abs(r).max()),1e-12)
    print(f"[{'PASS' if e/sc<2e-5 else 'FAIL'}] {name:12s} max|d|={e:.3e} rel={e/sc:.3e}")

pos=T("in_pos"); nb=T("current_neighbours"); resi=T("in_resi"); chain=T("in_chain")
batch=T("in_batch"); mask=T("in_mask")
idx=torch.arange(pos.shape[0])
pc={k: v[idx[:,None], nb] for k,v in dict(dmap=T("in_dmap"),dmap_mask=T("in_dmap_mask"),flags=T("in_flags")).items()}

f_relpos = GF.sequence_relative_position(32, one_hot=True, cyclic=False)(resi,chain,batch,nb)
f_dmap = torch.where(pc["dmap_mask"][...,None], GF.distance_rbf(pc["dmap"]), torch.zeros(1))
f_flags = torch.where(pc["flags"].any(dim=-1)[...,None], pc["flags"], torch.zeros_like(pc["flags"]))
f_dist = GF.distance_features(pos, nb, 0.0, 22.0)
f_dir  = GF.direction_features(pos, nb)
f_rot  = GF.position_rotation_features(pos, nb)
f_pv   = GF.pair_vector_features(pos, nb)
for n, g in [("f_relpos",f_relpos),("f_dmap",f_dmap),("f_flags",f_flags),
             ("f_dist",f_dist),("f_dir",f_dir),("f_rot",f_rot),("f_pairvec",f_pv)]:
    rep(n, g, R[n])

names=["linear","linear_1","linear_2","linear_3","linear_4","linear_5","linear_6"]
feats=[f_relpos,f_dmap,f_flags,f_dist,f_dir,f_rot,f_pv]
acc=torch.zeros(pos.shape[0], nb.shape[1], 64)
for i,(nm,f) in enumerate(zip(names,feats)):
    w=torch.from_numpy(NPZ[P+nm+"/w"][L].copy())
    acc = acc + torch.einsum("...k,ki->...i", f, w)
    rep(f"acc_{i}", acc, R[f"acc_{i}"])

ln=HkLayerNorm(64); ln.scale.copy_(torch.from_numpy(NPZ[P+"layer_norm/scale"][L].copy()))
ln.offset.copy_(torch.from_numpy(NPZ[P+"layer_norm/offset"][L].copy()))
a=ln(acc); rep("after_ln", a, R["after_ln"])
m=MLP(64,128,out_size=64,depth=2,activation=gelu,final_init="linear")
m.layers[0].w.copy_(torch.from_numpy(NPZ[P+"mlp/linear/w"][L].copy()))
m.layers[0].b.copy_(torch.from_numpy(NPZ[P+"mlp/linear/b"][L].copy()))
m.layers[1].w.copy_(torch.from_numpy(NPZ[P+"mlp/linear_1/w"][L].copy()))
m.layers[1].b.copy_(torch.from_numpy(NPZ[P+"mlp/linear_1/b"][L].copy()))
rep("after_mlp", m(a), R["after_mlp"])
