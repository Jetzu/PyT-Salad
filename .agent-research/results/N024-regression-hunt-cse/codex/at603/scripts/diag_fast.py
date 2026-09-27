"""Is the compiled/eager gap TF32, or a real divergence? And where does time go?"""
import os,sys,time
import torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
from salad_torch.modules.diffusion import uniform_draw_count
dev=torch.device("cuda"); cfg=get_config("default_vp")
m=StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
L=100; A=20; z=torch.zeros(L,dtype=torch.long,device=dev)
pos=torch.randn(L,A,3,device=dev)*4
d=dict(pos=pos,pos_noised=pos,aa_gt=torch.full((L,),20,dtype=torch.long,device=dev),
  seq=torch.full((L,),20,dtype=torch.long,device=dev),seq_noised=torch.full((L,),20,dtype=torch.long,device=dev),
  residue_index=torch.arange(L,device=dev),chain_index=z,batch_index=z,
  mask=torch.ones(L,dtype=torch.bool,device=dev),cyclic_mask=torch.zeros(L,dtype=torch.bool,device=dev),
  t_pos=torch.full((L,),0.5,device=dev),t_seq=torch.ones(L,device=dev))
p={"pos":torch.zeros(L,A,3,device=dev),"local":torch.zeros(L,cfg.local_size,device=dev)}
U=torch.rand(uniform_draw_count(cfg),L,L,device=dev).clamp_(1e-6,1-1e-6)
c=m.prepare_condition(d); pc=c["pair_condition"]
def core():
    return m.diffusion.forward_core(d["pos_noised"],p["pos"],p["local"],c["condition"],
        pc["dmap"],pc["omap"],pc["dmap_mask"],pc["flags"],d["t_pos"],d["t_seq"],
        d["seq_noised"],d["aa_gt"],d["pos"],d["residue_index"],d["chain_index"],
        d["batch_index"],d["mask"],None,U)
for prec in ("highest","high"):
    torch.set_float32_matmul_precision(prec)
    with torch.no_grad():
        ref=core()
        fast=FastDenoiser(m,L,device=dev,mode="reduce-overhead").warmup()
        got=fast(d,p,uniform=U)[0]
    dp=float((got["pos"]-ref[0]).norm(dim=-1).max())
    na=int((got["aatype"]!=ref[10]).sum())
    print(f"matmul_precision={prec:<8} compiled-vs-eager: max|dpos|={dp:.3e} A  aatype diff {na}/{L}")
# eager fp32 vs eager tf32, to size the TF32 effect on its own
torch.set_float32_matmul_precision("highest")
with torch.no_grad(): a=core()[0]
torch.set_float32_matmul_precision("high")
with torch.no_grad(): b=core()[0]
print(f"eager highest-vs-high (pure TF32 effect): max|dpos|={float((a-b).norm(dim=-1).max()):.3e} A")
