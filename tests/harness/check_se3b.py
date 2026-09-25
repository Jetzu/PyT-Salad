import sys
import numpy as np, torch
sys.path.insert(0,"/home/hpc/v181bb/v181bb10/SALAD_torch")
torch.set_grad_enabled(False)
SC="/tmp/claude-215432/-home-hpc-v181bb-v181bb10/0eeb4a3c-fee2-4b4d-8ef9-9555175d051e/scratchpad/"
D=np.load(SC+"block_ref.npz")
def T(k):
    a=D[k]
    if a.dtype==np.bool_: return torch.from_numpy(a)
    if np.issubdtype(a.dtype,np.integer): return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a.astype(np.float32))
import torch.nn.functional as F
import salad_torch.modules.basic as B, salad_torch.modules.blocks as BL
tg=lambda x: F.gelu(x, approximate="tanh"); B.gelu=tg; BL.gelu=tg
class dd(dict):
    __getattr__=dict.get; __setattr__=dict.__setitem__
cfg=dd(local_size=128,pair_size=64,resi_dual=False,factor=4,heads=8,key_size=32,
       diffusion_depth=6,sigma_data=10.0,preconditioning="vp")
from salad_torch.modules.blocks import DiffusionBlock
blk=DiffusionBlock(cfg,128,20,False).eval()
gen=torch.Generator().manual_seed(11)
for p in blk.parameters(): p.copy_(0.2*torch.randn(p.shape, generator=gen))
pc=dict(dmap=T("in_dmap"),dmap_mask=T("in_dmap_mask"),flags=T("in_flags"))
base=[T("in_features"),None,None,T("in_condition"),pc,T("in_time"),
      T("in_resi"),T("in_chain"),T("in_batch"),T("in_mask"),
      T("current_neighbours"),T("cond_neighbours")]
def run(p,pp):
    a=list(base); a[1]=p; a[2]=pp; return blk(*a)
pos=T("in_pos"); prev=T("in_prev_pos")
pos = pos - pos[:,1].mean(0)      # centre, so translations are the only offset
prev = prev - prev[:,1].mean(0)
f0,p0 = run(pos,prev)
sc_f=float(f0.abs().max()); sc_p=float(p0.abs().max())

def report(tag, Q, tv):
    xf=lambda x: torch.einsum("ij,...j->...i",Q,x)+tv
    f1,p1=run(xf(pos),xf(prev))
    ei=float((f1-f0).abs().max()); ee=float((p1-xf(p0)).abs().max())
    print(f"  {tag:32s} inv {ei:9.3e} ({ei/sc_f:8.2e} rel)   equi {ee:9.3e} ({ee/sc_p:8.2e} rel)")

I3=torch.eye(3)
Rz90=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
report("exact 90deg rot, t=0", Rz90, torch.zeros(3))
report("identity rot, t=(0,0,0)", I3, torch.zeros(3))
report("identity rot, t=(8,8,8)", I3, torch.full((3,),8.0))
report("identity rot, t=(1000,0,0)", I3, torch.tensor([1000.,0.,0.]))
g=torch.Generator().manual_seed(5)
A=torch.randn(3,3,generator=g,dtype=torch.float64); Q,_=torch.linalg.qr(A)
if torch.det(Q)<0: Q[:,0]=-Q[:,0]
report("generic rot, t=0", Q.float(), torch.zeros(3))
report("generic rot, t=(13,-27,4)", Q.float(), torch.tensor([13.,-27.,4.]))
