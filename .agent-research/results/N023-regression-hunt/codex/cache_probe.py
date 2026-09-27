"""P7 cache lifecycle on CPU; no repository source changes."""
import sys
sys.path.insert(0, '/home/hpc/v181bb/v181bb10/SALAD_torch')
import torch
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
c=get_config('default_vp')
m=StructureDiffusionPredict(c).eval(); s=m.diffusion.diffusion_stack
print('init_sync',s.cond_pair_stack.in_sync(s.blocks))
with torch.no_grad(): s.blocks[0].pair_2.linear_dist.w.data.add_(0.125)
print('manual_edit_sync',s.cond_pair_stack.in_sync(s.blocks),'delta',float((s.blocks[0].pair_2.linear_dist.w-s.cond_pair_stack.w_dist[0]).abs().max()))
m.load_state_dict(m.state_dict())
print('post_load_state_dict_sync',s.cond_pair_stack.in_sync(s.blocks))
m=m.to(torch.float64); s=m.diffusion.diffusion_stack
print('post_to_float64_sync',s.cond_pair_stack.in_sync(s.blocks))
m2=StructureDiffusionPredict(c).eval(); s2=m2.diffusion.diffusion_stack
print('second_model_sync',s2.cond_pair_stack.in_sync(s2.blocks))
load_model(m2,load_param_npz('/hnvme/workspace/v181bb10-salad/default_vp-200k.npz'),c,strict=True)
print('post_converter_sync',s2.cond_pair_stack.in_sync(s2.blocks))
f=FastDenoiser(m2,100,device='cpu',capture=False,preset='fast')
print('post_denoiser_sync',s2.cond_pair_stack.in_sync(s2.blocks))
with torch.no_grad(): s2.blocks[0].pair_2.linear_dist.w.data.add_(0.125)
print('post_denoiser_manual_edit_sync',s2.cond_pair_stack.in_sync(s2.blocks))
