"""Top kernels in the compiled forward, plus the TF32 question."""
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
def bench(fn,it=30,warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3

for prec,mode in (("highest","reduce-overhead"),("high","reduce-overhead"),("high","max-autotune")):
    torch.set_float32_matmul_precision(prec)
    torch._dynamo.reset()
    with torch.no_grad():
        f=FastDenoiser(m,L,device=dev,mode=mode).warmup()
        t=bench(lambda: f(d,p))
    print(f"precision={prec:<8} mode={mode:<16} {t:6.2f} ms   500-step {t*500/1000:5.2f} s")

# kernel breakdown at the default (correct) precision
torch.set_float32_matmul_precision("highest"); torch._dynamo.reset()
with torch.no_grad():
    f=FastDenoiser(m,L,device=dev,mode="reduce-overhead").warmup()
    for _ in range(5): f(d,p)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(10): f(d,p)
        torch.cuda.synchronize()
print("\n=== top CUDA kernels (10 forwards) ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18,
                                max_name_column_width=60))
