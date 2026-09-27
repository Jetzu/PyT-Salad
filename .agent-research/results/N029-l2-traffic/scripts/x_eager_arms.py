import sys, numpy as np, torch
sys.path.insert(0,'.')
from salad_torch import opt
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.geometry import features as GF
torch.set_grad_enabled(False)
import os
CFG=os.environ.get("CFG","default_vp")
D=np.load(f"fixtures/model_{CFG}.npz", allow_pickle=False)
cfg=get_config(CFG)
def _t(a):
    a=np.asarray(a)
    if a.dtype==np.bool_: return torch.from_numpy(a)
    if np.issubdtype(a.dtype,np.integer): return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a.astype(np.float32))
m=StructureDiffusionPredict(cfg).eval()
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{CFG}-200k.npz"), cfg, strict=True)
data={k[3:]:_t(D[k]) for k in D.files if k.startswith("in_") and k not in ("in_prev_pos","in_prev_local")}
for k in ("pos_noised","seq_noised","t_pos","t_seq"): data[k]=_t(D[f"noised_{k}"])
prev={"pos":_t(D["in_prev_pos"]),"local":_t(D["in_prev_local"])}
uni=_t(D["fixed_uniform"])
calls={}
orig=GF.pair_vector_features
def wrapped(pos,nb,scale=0.1):
    r=orig(pos,nb,scale); calls.setdefault(opt.PAIRVEC_SHARE,[]).append(r.clone()); return r
GF.pair_vector_features=wrapped
import salad_torch.modules.blocks as B; B.GF.pair_vector_features=wrapped
res={}
for arm in (0,1):
    opt.PAIRVEC_SHARE=bool(arm)
    r,np_=m(data,prev,predict_aa=True,uniform=uni)
    res[arm]={k:v.clone() for k,v in r.items() if torch.is_tensor(v)}
a,b=calls[False],calls[True]
print("pairvec calls per arm:",len(a),len(b))
bad=[i for i,(x,y) in enumerate(zip(a,b)) if not torch.equal(x,y)]
print("pairvec outputs differing:",len(bad),"of",len(a), bad[:5])
if bad:
    i=bad[0]; print("  first:",(a[i].double()-b[i].double()).abs().max().item(), a[i].shape)
for k in res[0]:
    x,y=res[0][k],res[1][k]
    if x.is_floating_point():
        print(f"{k:14s} bitwise={torch.equal(x,y)} max|d|={(x.double()-y.double()).abs().max().item():.3e}")
