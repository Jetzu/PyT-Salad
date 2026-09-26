import os,sys,statistics,collections,time
import torch
from torch.profiler import profile,ProfilerActivity
sys.path.insert(0,os.getcwd())
from salad_torch.config import get_config
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
from salad_torch.convert.haiku_to_torch import load_model,load_param_npz
D='cuda'; torch.set_float32_matmul_precision('highest')
print('torch',torch.__version__,'gpu',torch.cuda.get_device_name(),'cache',os.environ.get('TORCHINDUCTOR_CACHE_DIR'),flush=True)

def event_bench(fn,reps=300):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(reps): fn()
    en.record(); en.synchronize()
    return st.elapsed_time(en)*1000/reps

def graph_bench(fn,reps=1000):
    for _ in range(10): fn()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return event_bench(g.replay,reps)

for k in (400,240,60,9):
    M=100*48; N=64; B=6
    x=torch.randn(M,k,device=D); w=torch.randn(k,N,device=D)
    xb=torch.randn(B,M,k,device=D); wb=torch.randn(B,k,N,device=D)
    y=torch.empty(M,N,device=D); yb=torch.empty(B,M,N,device=D)
    one=lambda: torch.mm(x,w,out=y)
    six=lambda: [torch.mm(xb[i],wb[i],out=yb[i]) for i in range(B)]
    batch=lambda: torch.bmm(xb,wb,out=yb)
    print('GEMM',M,k,N,'mm_launch_us',event_bench(one),'mm_graphed_us',graph_bench(one),'six_graphed_us',graph_bench(six),'bmm_graphed_us',graph_bench(batch),flush=True)
    # Same inputs/weights to measure arithmetic reordering, not random samples.
    ref=torch.stack([xb[i]@wb[i] for i in range(B)])
    got=torch.bmm(xb,wb)
    print('BMM_DIFF',k,'max',float((ref-got).abs().max()),'equal',bool(torch.equal(ref,got)),flush=True)

cfg=get_config('default_vp'); m=StructureDiffusionPredict(cfg).eval().cuda()
load_model(m,load_param_npz('/hnvme/workspace/v181bb10-salad/default_vp-200k.npz'),cfg,strict=True)
f=FastDenoiser(m,100,device=D,manual_graph=True,mode='default',preset='exact')
f._static_in=f._blank_inputs()
with torch.no_grad(): f._build(False,False,iters=3)
g,st=f._graphs[(False,False)]; args=f._args(False,False)
for label,fn in [('captured',g.replay),('noncaptured',lambda:f._compiled(*args))]:
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5): fn()
        torch.cuda.synchronize()
    ev=[e for e in p.key_averages() if e.device_type.name=='CUDA' and e.device_time_total>0]
    n=sum(e.count for e in ev)//5; ms=sum(e.device_time_total for e in ev)/5000
    print('PROFILE',label,'kernels_per_call',n,'gpu_busy_ms',ms,flush=True)
    print('PROFILE_TOP',label,[(e.key,e.count//5,round(e.device_time_total/5000,4)) for e in sorted(ev,key=lambda z:-z.device_time_total)[:12]],flush=True)
print('REPLAY_US',graph_bench(g.replay,reps=500),flush=True)
