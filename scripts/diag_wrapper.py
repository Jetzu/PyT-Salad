"""Where does FastDenoiser's wrapper overhead go, and why only at L>=100?"""
import os,sys,time,torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser, _CORE_ARGS
from torch._dynamo.utils import counters
dev=torch.device("cuda"); cfg=get_config("default_vp")
m=StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
def mk(L):
    A=20; z=torch.zeros(L,dtype=torch.long,device=dev); pos=torch.randn(L,A,3,device=dev)*4
    d=dict(pos=pos,pos_noised=pos,aa_gt=torch.full((L,),20,dtype=torch.long,device=dev),
      seq=torch.full((L,),20,dtype=torch.long,device=dev),seq_noised=torch.full((L,),20,dtype=torch.long,device=dev),
      residue_index=torch.arange(L,device=dev),chain_index=z,batch_index=z,
      mask=torch.ones(L,dtype=torch.bool,device=dev),cyclic_mask=torch.zeros(L,dtype=torch.bool,device=dev),
      t_pos=torch.full((L,),0.5,device=dev),t_seq=torch.ones(L,device=dev))
    p={"pos":torch.zeros(L,A,3,device=dev),"local":torch.zeros(L,128,device=dev)}
    return d,p
def bench(fn,it=20,warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
for L in (50,100,200):
    d,p=mk(L); counters.clear()
    with torch.no_grad():
        fast=FastDenoiser(m,L,device=dev,mode="reduce-overhead").warmup()
        whole = bench(lambda: fast(d,p))
        # the compiled core alone, via the same static args the wrapper builds
        fast(d,p)
        args=[fast._static_in.get(k) for k in _CORE_ARGS]
        core = bench(lambda: fast._compiled(*args))
        cond = bench(lambda: m.prepare_condition(d))
        ind = dict(counters.get("inductor", {}))
    print(f"L={L:<5} whole {whole:7.2f} ms   core {core:7.2f} ms   overhead {whole-core:7.2f} ms"
          f"   (uncached prepare_condition would be {cond:6.2f} ms)")
    print(f"        cudagraph_skips={ind.get('cudagraph_skips')} "
          f"non_static_inputs={ind.get('cudagraph_recorded_non_static_inputs')} "
          f"unique_graphs={counters['stats'].get('unique_graphs')}")
