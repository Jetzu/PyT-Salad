"""N033 - lean/full step benchmark for the torch-upgrade evaluation.

Copy of N014's bench_p7.py with ONE change: ROOT comes from $SALAD_TREE (a
git-archive snapshot, per the hard rule to never import the live tree while a
sibling agent (N032) may be editing it), defaulting to the live tree only if
unset. Everything else - args, timing methodology (fresh process per call,
warmup then synchronize) - is unchanged so numbers are comparable to N014/N030.
"""
import argparse, os, sys, time, torch
ROOT = os.environ.get("SALAD_TREE", "/home/hpc/v181bb/v181bb10/SALAD_torch")
sys.path.insert(0, ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
from salad_torch import opt
ap=argparse.ArgumentParser(); ap.add_argument("--length",type=int,default=100)
ap.add_argument("--config",default="default_vp"); ap.add_argument("--iters",type=int,default=30)
ap.add_argument("--preset",default="exact"); ap.add_argument("--tag",default="")
ap.add_argument("--no-eager",action="store_true")
ap.add_argument("--chains",type=int,default=1)
a=ap.parse_args()

torch.set_float32_matmul_precision("highest")
dev=torch.device("cuda"); cfg=get_config(a.config)
m=StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"), cfg, strict=True)
L=a.length; A=5+cfg.augment_size; z=torch.zeros(L,dtype=torch.long,device=dev)
ch=(torch.arange(L,device=dev)*a.chains//L).long()
pos=torch.randn(L,A,3,device=dev)*4
d=dict(pos=pos,pos_noised=pos,aa_gt=torch.full((L,),20,dtype=torch.long,device=dev),
  seq=torch.full((L,),20,dtype=torch.long,device=dev),seq_noised=torch.full((L,),20,dtype=torch.long,device=dev),
  residue_index=torch.arange(L,device=dev),chain_index=ch,batch_index=z,
  mask=torch.ones(L,dtype=torch.bool,device=dev),cyclic_mask=torch.zeros(L,dtype=torch.bool,device=dev),
  t_pos=torch.full((L,),0.5,device=dev),t_seq=torch.ones(L,device=dev))
p={"pos":torch.zeros(L,A,3,device=dev),"local":torch.zeros(L,cfg.local_size,device=dev)}
def bench(fn,it,warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
with torch.no_grad():
    te = 0.0 if a.no_eager else bench(lambda: m(d,p,compute_distogram=False), a.iters)
    t_warm=time.perf_counter()
    fast=FastDenoiser(m,L,device=dev,preset=a.preset).warmup(
        pair_cond=(a.chains>1) or not opt.SKIP_PAIR_COND)
    t_warm=time.perf_counter()-t_warm
    tf=bench(lambda: fast(d,p), a.iters)
    tl=bench(lambda: fast(d,p,need_aa=False), a.iters) if fast.lean_steps else tf
    print(f"      gate: pair_cond_active={fast._pc_active}  graphs={sorted(fast._graphs)}")
print(f"RESULT {a.tag:<12} torch={torch.__version__:<14} L={L:<5} preset={a.preset:<8} "
      f"eager {te:7.2f}  full {tf:6.3f} ms  lean {tl:6.3f} ms   "
      f"500-step {(tf+499*tl)/1000:6.2f} s   compile {t_warm:6.1f} s")
