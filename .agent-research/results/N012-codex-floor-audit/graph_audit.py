import os, sys, shutil, argparse, collections
import torch
import torch._dynamo as dynamo
sys.path.insert(0, os.getcwd())
from salad_torch.config import get_config
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser

ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--length',type=int,default=48); a=ap.parse_args()
root=os.environ['TORCHINDUCTOR_CACHE_DIR']
print('torch',torch.__version__,'git branch config',a.config,'cache',root,flush=True)
cfg=get_config(a.config.replace('_cyclic',''))
if a.config.endswith('_cyclic'): cfg.cyclic=True
m=StructureDiffusionPredict(cfg).eval().cuda()
for preset in ('exact','fast'):
    f=FastDenoiser(m,a.length,device='cuda',preset=preset,manual_graph=True)
    f._static_in=f._blank_inputs()
    for need_aa in (False,True):
      for pc in (False,True):
        dynamo.reset(); shutil.rmtree(root,ignore_errors=True); os.makedirs(root,exist_ok=True)
        args=f._args(need_aa,pc)
        try:
          with torch.no_grad(): e=dynamo.explain(m.diffusion.forward_core)(*args)
          print('EXPLAIN',preset,need_aa,pc,'graphs',e.graph_count,'breaks',e.graph_break_count,'reasons',[str(x)[:160] for x in e.break_reasons],flush=True)
        except Exception as ex: print('ERROR',preset,need_aa,pc,repr(ex),flush=True)
