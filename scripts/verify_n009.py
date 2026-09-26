"""Verify two N009 claims first-hand: graph breaks, and CUDA determinism."""
import os,sys,torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.diffusion import uniform_draw_count
dev=torch.device("cuda"); cfg=get_config("default_vp"); L=48; A=20
m=StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
z=torch.zeros(L,dtype=torch.long,device=dev); pos=torch.randn(L,A,3,device=dev)*4
d=dict(pos=pos,pos_noised=pos,aa_gt=torch.full((L,),20,dtype=torch.long,device=dev),
  seq=torch.full((L,),20,dtype=torch.long,device=dev),seq_noised=torch.full((L,),20,dtype=torch.long,device=dev),
  residue_index=torch.arange(L,device=dev),chain_index=z,batch_index=z,
  mask=torch.ones(L,dtype=torch.bool,device=dev),cyclic_mask=torch.zeros(L,dtype=torch.bool,device=dev),
  t_pos=torch.full((L,),0.5,device=dev),t_seq=torch.ones(L,device=dev))
p={"pos":torch.zeros(L,A,3,device=dev),"local":torch.zeros(L,cfg.local_size,device=dev)}
U=torch.rand(uniform_draw_count(cfg),L,L,device=dev).clamp_(1e-6,1-1e-6)

# CLAIM 2: identical input, same process, twice -> should be 0 if deterministic
# eager takes ONE (N,N) field; the compiled core takes the (draws,N,N) stack
U2 = U[0]
with torch.no_grad():
    a=m(d,p,predict_aa=True,uniform=U2)[0]["pos"]
    b=m(d,p,predict_aa=True,uniform=U2)[0]["pos"]
print(f"CUDA eager, same input twice: max|d| = {float((a-b).abs().max()):.3e} A  (0 => deterministic)")

# CLAIM 1: graph breaks in the compiled region
import torch._dynamo as dynamo
from torch._dynamo.utils import counters
dynamo.reset(); counters.clear()
c=m.prepare_condition(d); pc=c["pair_condition"]
args=(d["pos_noised"],p["pos"],p["local"],c["condition"],pc["dmap"],pc["omap"],
      pc["dmap_mask"],pc["flags"],d["t_pos"],d["t_seq"],d["seq_noised"],d["aa_gt"],
      d["pos"],d["residue_index"],d["chain_index"],d["batch_index"],d["mask"],None,U)
comp=torch.compile(m.diffusion.forward_core, mode="default", dynamic=False)
with torch.no_grad():
    for _ in range(3): comp(*args)
gb=counters.get("graph_break",{})
print(f"graph_break counter: {sum(gb.values())} total, {len(gb)} distinct; "
      f"unique_graphs={counters['stats'].get('unique_graphs')}")
for r,n in sorted(gb.items(), key=lambda kv:-kv[1])[:5]: print(f"   {n}x {str(r)[:110]}")
