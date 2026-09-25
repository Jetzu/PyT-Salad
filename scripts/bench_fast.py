"""Diagnose the fast path: is it still launch-bound, and did cudagraphs engage?"""
import argparse, os, sys, time
import numpy as np, torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
from salad_torch.modules.diffusion import uniform_draw_count

ap=argparse.ArgumentParser()
ap.add_argument("--lengths",default="50,100,200,400")
ap.add_argument("--iters",type=int,default=30)
ap.add_argument("--mode",default="reduce-overhead")
ap.add_argument("--config",default="default_vp")
a=ap.parse_args()
torch.set_float32_matmul_precision("high")
dev=torch.device("cuda")
cfg=get_config(a.config)
print(f"device {torch.cuda.get_device_name(0)}  mode={a.mode}  tf32=high")

def mkdata(L):
    A=5+cfg.augment_size; z=torch.zeros(L,dtype=torch.long,device=dev)
    pos=torch.randn(L,A,3,device=dev)*4
    d=dict(pos=pos,pos_noised=pos,aa_gt=torch.full((L,),20,dtype=torch.long,device=dev),
      seq=torch.full((L,),20,dtype=torch.long,device=dev),seq_noised=torch.full((L,),20,dtype=torch.long,device=dev),
      residue_index=torch.arange(L,device=dev),chain_index=z,batch_index=z,
      mask=torch.ones(L,dtype=torch.bool,device=dev),cyclic_mask=torch.zeros(L,dtype=torch.bool,device=dev),
      t_pos=torch.full((L,),0.5,device=dev),t_seq=torch.ones(L,device=dev))
    p={"pos":torch.zeros(L,A,3,device=dev),"local":torch.zeros(L,cfg.local_size,device=dev)}
    return d,p
def bench(fn,iters,warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/iters*1e3

m=StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"), cfg, strict=True)
print(f"\n{'L':>6}{'eager ms':>11}{'fast ms':>10}{'speedup':>9}{'500-step s':>12}")
for L in [int(x) for x in a.lengths.split(",")]:
    d,p=mkdata(L)
    with torch.no_grad():
        te=bench(lambda: m(d,p,compute_distogram=False), a.iters)
        fast=FastDenoiser(m,L,device=dev,mode=a.mode).warmup()
        tf=bench(lambda: fast(d,p), a.iters)
    print(f"{L:>6}{te:>11.2f}{tf:>10.2f}{te/tf:>8.2f}x{tf*500/1000:>11.1f}")
# correctness with a SHARED uniform stack
L=100; d,p=mkdata(L)
U=torch.rand(uniform_draw_count(cfg),L,L,device=dev).clamp_(1e-6,1-1e-6)
with torch.no_grad():
    ref=m.diffusion.forward_core(
        d["pos_noised"],p["pos"],p["local"],
        m.prepare_condition(d)["condition"],
        *[m.prepare_condition(d)["pair_condition"][k] for k in ("dmap","omap","dmap_mask","flags")],
        d["t_pos"],d["t_seq"],d["seq_noised"],d["aa_gt"],d["pos"],
        d["residue_index"],d["chain_index"],d["batch_index"],d["mask"],None,U)
    fast=FastDenoiser(m,L,device=dev,mode=a.mode).warmup()
    got=fast(d,p,uniform=U)[0]
print(f"\nsame-uniform check: max|dpos| = {float((got['pos']-ref[0]).norm(dim=-1).max()):.2e} A  "
      f"aatype diff {int((got['aatype']!=ref[10]).sum())}/{L}")
try:
    from torch._inductor import cudagraph_trees
    print("inductor cudagraph containers:", len(getattr(cudagraph_trees,'get_container',lambda *a: {})(0) or {}) if False else "n/a")
except Exception: pass
