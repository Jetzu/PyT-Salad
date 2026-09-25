"""Final precision/speed sweep: what does it take to reach JAX?"""
import os,sys,time,torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
dev=torch.device("cuda"); cfg=get_config("default_vp"); L=100
m=StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
A=20; z=torch.zeros(L,dtype=torch.long,device=dev); pos=torch.randn(L,A,3,device=dev)*4
d=dict(pos=pos,pos_noised=pos,aa_gt=torch.full((L,),20,dtype=torch.long,device=dev),
  seq=torch.full((L,),20,dtype=torch.long,device=dev),seq_noised=torch.full((L,),20,dtype=torch.long,device=dev),
  residue_index=torch.arange(L,device=dev),chain_index=z,batch_index=z,
  mask=torch.ones(L,dtype=torch.bool,device=dev),cyclic_mask=torch.zeros(L,dtype=torch.bool,device=dev),
  t_pos=torch.full((L,),0.5,device=dev),t_seq=torch.ones(L,device=dev))
p={"pos":torch.zeros(L,A,3,device=dev),"local":torch.zeros(L,cfg.local_size,device=dev)}
def bench(fn,it=30,warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
with torch.no_grad():
    ref=m(d,p,compute_distogram=False)[0]["pos"]
for prec in ("highest","high"):
    torch.set_float32_matmul_precision(prec); torch._dynamo.reset()
    with torch.no_grad():
        f=FastDenoiser(m,L,device=dev,mode="reduce-overhead").warmup()
        t=bench(lambda: f(d,p)); dev_=float((f(d,p)[0]["pos"]-ref).norm(dim=-1).max())
    print(f"fp32 matmul={prec:<8} {t:5.2f} ms  500-step {t*500/1000:4.2f} s   max|dpos| vs fp32 eager {dev_:.2e} A")
torch.set_float32_matmul_precision("highest")
for dt in (torch.bfloat16, torch.float16):
    torch._dynamo.reset()
    with torch.no_grad():
        f=FastDenoiser(m,L,device=dev,mode="reduce-overhead").warmup()
        def run():
            with torch.autocast("cuda",dtype=dt): return f(d,p)
        run(); t=bench(run); dev_=float((run()[0]["pos"].float()-ref).norm(dim=-1).max())
    print(f"autocast {str(dt).split('.')[-1]:<11} {t:5.2f} ms  500-step {t*500/1000:4.2f} s   max|dpos| vs fp32 eager {dev_:.2e} A")
print("\nJAX reference: 4.73 ms fp32 / 3.95 ms default(TF32)")
