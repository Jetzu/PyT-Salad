import os,sys
import torch
from torch.profiler import profile,ProfilerActivity
sys.path.insert(0,os.getcwd())
from salad_torch.config import get_config
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
from salad_torch.convert.haiku_to_torch import load_model,load_param_npz
cfg=get_config('default_vp');m=StructureDiffusionPredict(cfg).eval().cuda()
load_model(m,load_param_npz('/hnvme/workspace/v181bb10-salad/default_vp-200k.npz'),cfg,strict=True)
f=FastDenoiser(m,100,device='cuda',manual_graph=True,mode='default',preset='exact')
f._static_in=f._blank_inputs()
with torch.no_grad():
 f._build(False,False,iters=3)
 g,st=f._graphs[(False,False)]; args=f._args(False,False)
 for label,fn in [('captured',g.replay),('noncaptured',lambda:f._compiled(*args))]:
  for _ in range(3): fn()
  torch.cuda.synchronize()
  with profile(activities=[ProfilerActivity.CUDA]) as p:
   for _ in range(5): fn()
   torch.cuda.synchronize()
  ev=[e for e in p.key_averages() if e.device_type.name=='CUDA' and e.device_time_total>0]
  print('PROFILE',label,'kernels',sum(e.count for e in ev)//5,'gpu_busy_ms',sum(e.device_time_total for e in ev)/5000,flush=True)
  print('COUNTS',label,{k:sum(e.count for e in ev if k in e.key.lower())//5 for k in ('triton','cublas','memcpy')},flush=True)
 stt=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
 stt.record()
 for _ in range(200):g.replay()
 end.record();end.synchronize()
 print('REPLAY_EVENT_US',stt.elapsed_time(end)*1000/200,flush=True)
