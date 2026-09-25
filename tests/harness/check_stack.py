import sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/home/hpc/v181bb/v181bb10/SALAD_torch")
torch.set_grad_enabled(False)
SC="/tmp/claude-215432/-home-hpc-v181bb-v181bb10/0eeb4a3c-fee2-4b4d-8ef9-9555175d051e/scratchpad/"
D=np.load(SC+"block_ref.npz"); S=np.load(SC+"stack_ref.npz")
NPZ=np.load("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
P="structure_diffusion/diffusion/diffusion_stack/__layer_stack_with_state/diffusion_block/"
import salad_torch.modules.basic as B, salad_torch.modules.blocks as BL
from salad_torch.modules.basic import MLP
class dd(dict):
    __getattr__=dict.get; __setattr__=dict.__setitem__
cfg=dd(local_size=128,pair_size=64,resi_dual=False,factor=4,heads=8,key_size=32,
       diffusion_depth=6,sigma_data=10.0,preconditioning="vp")
def T(k,src=D):
    a=src[k]
    if a.dtype==np.bool_: return torch.from_numpy(a)
    if np.issubdtype(a.dtype,np.integer): return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a.astype(np.float32))
MAP={}
def lin(p,h): MAP[p]=("lin",h)
def ln(p,h): MAP[p]=("ln",h)
for i,n in enumerate(["relpos","dmap","flags","dist","dir","rot","pairvec"]):
    lin(f"pair_1.linear_{n}","linear" if i==0 else f"linear_{i}")
    lin(f"pair_2.linear_{n}",f"linear_{7+i}")
ln("pair_1.layer_norm","layer_norm"); ln("pair_2.layer_norm","layer_norm_2")
lin("pair_1.mlp.layers.0","mlp/linear"); lin("pair_1.mlp.layers.1","mlp/linear_1")
lin("pair_2.mlp.layers.0","mlp_1/linear"); lin("pair_2.mlp.layers.1","mlp_1/linear_1")
for x in ("1","3","4","5"): ln(f"layer_norm_{x}",f"layer_norm_{x}")
for pa,ha in (("attn_1","sparse_structure_attn"),("attn_2","sparse_structure_attn_1")):
    b=f"{pa}.ada_point_attention."; h=f"{ha}/ada_point_attention/"
    for s in ("layer_norm","layer_norm_1","layer_norm_2"): ln(b+s,h+s)
    for s in ("qkv","bias","project_out"): lin(b+s,h+s)
    lin(b+"qkv_global.linear",h+"qkv_global/linear"); MAP[b+"gamma"]=("g",h+"gamma")
for p,h in (("linear_condition","linear"),("mlp.layers.0","mlp/linear"),
            ("mlp.layers.1","mlp/linear_1"),("linear_update","linear_1"),
            ("linear_gate","linear_2"),("linear_chain_gate","linear_3"),
            ("linear_batch_gate","linear_4"),("linear_out","linear_5")):
    lin("update."+p,"light_global_update/"+h)
lin("update_positions.linear","linear_14")

def build(tanh):
    if tanh:
        g=lambda x: F.gelu(x,approximate="tanh")
    else:
        g=F.gelu
    B.gelu=g; BL.gelu=g
    from salad_torch.modules.blocks import DiffusionStack
    st=DiffusionStack(cfg,128,20,False).eval()
    for m in st.modules():
        if isinstance(m,MLP) and m.activation is not F.relu: m.activation=g
    sd={}
    for L in range(6):
        for port,(kind,h) in MAP.items():
            k=f"blocks.{L}."+port
            if kind=="lin":
                sd[k+".w"]=torch.from_numpy(NPZ[P+h+"/w"][L].copy())
                sd[k+".b"]=torch.from_numpy(NPZ[P+h+"/b"][L].copy())
            elif kind=="ln":
                sd[k+".scale"]=torch.from_numpy(NPZ[P+h+"/scale"][L].copy())
                sd[k+".offset"]=torch.from_numpy(NPZ[P+h+"/offset"][L].copy())
            else:
                sd[k]=torch.from_numpy(NPZ[P+h][L].copy())
    miss,unex=st.load_state_dict(sd,strict=True)
    return st
pc=dict(dmap=T("in_dmap"),dmap_mask=T("in_dmap_mask"),flags=T("in_flags"))
cur=torch.from_numpy(S["cur"].astype(np.int64)); cond=torch.from_numpy(S["cond"].astype(np.int64))
step=[0]
def nf(p,pp):
    i=step[0]; step[0]+=1; return cur[i],cond[i]
for tanh in (False,True):
    st=build(tanh); step[0]=0
    l,p,tr=st(T("in_features"),T("in_pos"),T("in_prev_pos"),T("in_condition"),pc,
              T("in_time"),T("in_resi"),T("in_chain"),T("in_batch"),T("in_mask"),nf)
    for nm,g,r in (("local",l.numpy(),S["out_local"]),("pos",p.numpy(),S["out_pos"]),
                   ("traj",tr.numpy(),S["traj"])):
        e=float(np.abs(g-r).max()); sc=float(np.abs(r).max())
        print(f"  gelu={'tanh' if tanh else 'erf '}  {nm:6s} max|d|={e:.3e} rel={e/sc:.2e}")
    # per-atom RMSD of the final coordinates
    d=np.sqrt(((p.numpy()-S['out_pos'])**2).sum(-1))
    print(f"        final-coord mean dev = {d.mean():.4f} A, max = {d.max():.4f} A")
