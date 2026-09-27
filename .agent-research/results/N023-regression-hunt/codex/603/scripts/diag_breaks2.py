"""Real graph-break + cudagraph evidence for forward_core."""
import os,sys,time,torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.diffusion import uniform_draw_count
dev=torch.device("cuda"); cfg=get_config("default_vp")
m=StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
L=100; A=20; z=torch.zeros(L,dtype=torch.long,device=dev)
pos=torch.randn(L,A,3,device=dev)*4
base=dict(pos=pos,pos_noised=pos,aa_gt=torch.full((L,),20,dtype=torch.long,device=dev),
  seq=torch.full((L,),20,dtype=torch.long,device=dev),seq_noised=torch.full((L,),20,dtype=torch.long,device=dev),
  residue_index=torch.arange(L,device=dev),chain_index=z,batch_index=z,
  mask=torch.ones(L,dtype=torch.bool,device=dev),cyclic_mask=torch.zeros(L,dtype=torch.bool,device=dev),
  t_pos=torch.full((L,),0.5,device=dev),t_seq=torch.ones(L,device=dev))
p={"pos":torch.zeros(L,A,3,device=dev),"local":torch.zeros(L,128,device=dev)}
U=torch.rand(uniform_draw_count(cfg),L,L,device=dev).clamp_(1e-6,1-1e-6)
c=m.prepare_condition(base); pc=c["pair_condition"]
args=(base["pos_noised"],p["pos"],p["local"],c["condition"],pc["dmap"],pc["omap"],
      pc["dmap_mask"],pc["flags"],base["t_pos"],base["t_seq"],base["seq_noised"],
      base["aa_gt"],base["pos"],base["residue_index"],base["chain_index"],
      base["batch_index"],base["mask"],None,U)

import torch._dynamo as dynamo
from torch._dynamo.utils import counters
dynamo.reset(); counters.clear()
comp = torch.compile(m.diffusion.forward_core, mode="reduce-overhead", dynamic=False)
with torch.no_grad():
    for _ in range(4): out = comp(*args)
torch.cuda.synchronize()

gb = counters.get("graph_break", {})
print(f"\n=== GRAPH BREAKS: {sum(gb.values())} total, {len(gb)} distinct ===")
for reason, n in sorted(gb.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {n:>4}x  {str(reason)[:160]}")
print(f"\nunique_graphs: {counters['stats'].get('unique_graphs')}")

# did cudagraphs engage?
try:
    from torch._inductor.cudagraph_trees import get_manager
    mgr = get_manager(0, create_if_none_exists=False)
    print(f"cudagraph manager present: {mgr is not None}")
    if mgr is not None:
        print(f"  recorded cudagraph nodes: {len(getattr(mgr,'roots',{}) or {})}")
except Exception as e:
    print("cudagraph manager query failed:", type(e).__name__, e)
try:
    print("inductor cudagraph skip reasons:", dict(counters.get("inductor", {})))
except Exception: pass

def bench(fn,it=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
with torch.no_grad():
    print(f"\ncompiled forward_core: {bench(lambda: comp(*args)):.2f} ms")
    print(f"eager    forward_core: {bench(lambda: m.diffusion.forward_core(*args)):.2f} ms")
