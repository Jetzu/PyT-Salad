import os,sys,copy
import torch
import torch._dynamo as dynamo
from torch._dynamo.utils import counters
sys.path.insert(0,os.getcwd())
from salad_torch.config import get_config
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser

dynamo.reset(); counters.clear()
cfg=get_config('default_vp'); m=StructureDiffusionPredict(cfg).eval().cuda()
f=FastDenoiser(m,48,device='cuda',manual_graph=False,capture=False,mode='default',preset='exact')
f._static_in=f._blank_inputs()
with torch.no_grad():
 for aa in (False,True):
  for pc in (False,True):
   args=f._args(aa,pc)
   f._compiled(*args)
   before={k:dict(v) for k,v in counters.items()}
   for i in range(10): f._compiled(*args)
   after={k:dict(v) for k,v in counters.items()}
   print('VARIANT',aa,pc,'BEFORE',before,'AFTER',after,flush=True)
 print('ALL_VARIANTS', {k:dict(v) for k,v in counters.items()},flush=True)
