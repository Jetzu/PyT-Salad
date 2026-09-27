"""List every torch.compile graph break in forward_core. Graph breaks disable cudagraphs."""
import os,sys,torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.diffusion import uniform_draw_count
import torch._dynamo as dynamo
dev=torch.device("cuda"); cfg=get_config("default_vp")
m=StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
L=100; A=20; z=torch.zeros(L,dtype=torch.long,device=dev)
pos=torch.randn(L,A,3,device=dev)*4
d=dict(pos=pos,pos_noised=pos,aa_gt=torch.full((L,),20,dtype=torch.long,device=dev),
  seq_noised=torch.full((L,),20,dtype=torch.long,device=dev),
  residue_index=torch.arange(L,device=dev),chain_index=z,batch_index=z,
  mask=torch.ones(L,dtype=torch.bool,device=dev),
  t_pos=torch.full((L,),0.5,device=dev),t_seq=torch.ones(L,device=dev))
p={"pos":torch.zeros(L,A,3,device=dev),"local":torch.zeros(L,128,device=dev)}
U=torch.rand(uniform_draw_count(cfg),L,L,device=dev).clamp_(1e-6,1-1e-6)
c=m.prepare_condition(dict(d, pos=pos, seq=d["seq_noised"], cyclic_mask=torch.zeros(L,dtype=torch.bool,device=dev)))
pc=c["pair_condition"]
args=(d["pos_noised"],p["pos"],p["local"],c["condition"],pc["dmap"],pc["omap"],
      pc["dmap_mask"],pc["flags"],d["t_pos"],d["t_seq"],d["seq_noised"],d["aa_gt"],
      d["pos"],d["residue_index"],d["chain_index"],d["batch_index"],d["mask"],None,U)
dynamo.reset()
expl = dynamo.explain(m.diffusion.forward_core)(*args)
print(f"graph count      : {expl.graph_count}")
print(f"graph break count: {expl.graph_break_count}")
print(f"op count         : {expl.op_count}")
for i,r in enumerate(expl.break_reasons[:15]):
    rsn = getattr(r,'reason',r)
    print(f"  break {i}: {str(rsn)[:150]}")
