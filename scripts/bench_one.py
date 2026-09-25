"""Benchmark ONE length in a fresh process — compiling several shapes in one
process degrades inductor's cudagraphs, which made earlier numbers look flat."""
import argparse,os,sys,time,torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
ap=argparse.ArgumentParser(); ap.add_argument("--length",type=int,default=100)
ap.add_argument("--config",default="default_vp"); ap.add_argument("--iters",type=int,default=30)
a=ap.parse_args()
dev=torch.device("cuda"); cfg=get_config(a.config)
m=StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"), cfg, strict=True)
L=a.length; A=5+cfg.augment_size; z=torch.zeros(L,dtype=torch.long,device=dev)
pos=torch.randn(L,A,3,device=dev)*4
d=dict(pos=pos,pos_noised=pos,aa_gt=torch.full((L,),20,dtype=torch.long,device=dev),
  seq=torch.full((L,),20,dtype=torch.long,device=dev),seq_noised=torch.full((L,),20,dtype=torch.long,device=dev),
  residue_index=torch.arange(L,device=dev),chain_index=z,batch_index=z,
  mask=torch.ones(L,dtype=torch.bool,device=dev),cyclic_mask=torch.zeros(L,dtype=torch.bool,device=dev),
  t_pos=torch.full((L,),0.5,device=dev),t_seq=torch.ones(L,device=dev))
p={"pos":torch.zeros(L,A,3,device=dev),"local":torch.zeros(L,cfg.local_size,device=dev)}
def bench(fn,it,warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
with torch.no_grad():
    te=bench(lambda: m(d,p,compute_distogram=False), a.iters)
    fast=FastDenoiser(m,L,device=dev,mode="reduce-overhead").warmup()
    tf=bench(lambda: fast(d,p), a.iters)
print(f"RESULT L={L:<5} eager {te:7.2f} ms   fast {tf:6.2f} ms   {te/tf:5.2f}x   "
      f"500-step {tf*500/1000:6.2f} s")
