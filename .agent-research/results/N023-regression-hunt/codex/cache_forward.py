"""Demonstrate that post-construction parameter edits leave P7 cache stale."""
import sys
sys.path.insert(0,'/home/hpc/v181bb/v181bb10/SALAD_torch')
import torch
from salad_torch.config import get_config
from salad_torch.modules.model import StructureDiffusionPredict
c=get_config('default_vp'); torch.manual_seed(2)
m=StructureDiffusionPredict(c).eval(); s=m.diffusion.diffusion_stack
L,A,B,K=8,5+c.augment_size,c.diffusion_depth,4
pos=torch.randn(L,A,3); resi=torch.arange(L); chain=torch.zeros(L,dtype=torch.long); batch=chain.clone(); mask=torch.ones(L,dtype=torch.bool)
nb=torch.arange(K).repeat(B,L,1)
with torch.no_grad():
 p0=s.cond_pair_stack(pos,None,nb,resi,chain,batch,mask)[0][0]
 q0=s.blocks[0].pair_2(pos,None,nb[0],resi,chain,batch,mask)[0]
 s.blocks[0].pair_2.mlp.layers[1].b.data[0]+=5.0
 p1=s.cond_pair_stack(pos,None,nb,resi,chain,batch,mask)[0][0]
 q1=s.blocks[0].pair_2(pos,None,nb[0],resi,chain,batch,mask)[0]
 print('cache_sync_after_edit',s.cond_pair_stack.in_sync(s.blocks))
 print('batched_after_edit_max_abs',float((p1-p0).abs().max()))
 print('sequential_after_edit_max_abs',float((q1-q0).abs().max()))
 s.cond_pair_stack.sync(s.blocks)
 p2=s.cond_pair_stack(pos,None,nb,resi,chain,batch,mask)[0][0]
 print('batched_after_sync_max_abs',float((p2-p0).abs().max()))
