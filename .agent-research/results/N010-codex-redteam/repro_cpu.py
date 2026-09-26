import numpy as np, torch
from pathlib import Path
from salad_torch.modules.fast import FastDenoiser
from salad_torch.inference.symmetry import parse_sym, screw_replicate
from salad_torch.losses.structure import base_weight
from salad_torch.losses.sequence import aa_loss, dssp_loss
from salad_torch.io.pdb import Protein, to_pdb
class C:
    local_size=2
    cyclic=False
    multi_motif=False
    diffusion_depth=1
class D:
    atom_count=5
    def forward_core(self,*a):
        c=a[3]; n=len(c); p=c[:,:1,None].expand(-1,5,3).clone()
        return (p,c,p[None],torch.zeros(n,20),c,torch.zeros(n,3),torch.zeros(n,1,dtype=torch.long),c,c,p[:,:1],torch.zeros(n,dtype=torch.long))
class M:
    config=C(); diffusion=D()
    def prepare_condition(self,d):
        n=len(d['dssp_condition']); c=d['dssp_condition'].float()[:,None].expand(-1,2).clone()
        return {'condition':c,'pair_condition':dict(dmap=torch.zeros(n,n),omap=torch.zeros(n,n,9),dmap_mask=torch.zeros(n,n,dtype=torch.bool),flags=torch.zeros(n,n,3))}
m=M(); f=FastDenoiser(m,4,device='cpu',capture=False); f._compiled=m.diffusion.forward_core
z=torch.zeros(4,dtype=torch.long)
d=dict(pos_noised=torch.zeros(4,5,3),pos=torch.zeros(4,5,3),t_pos=torch.ones(4),t_seq=torch.ones(4),seq_noised=z,aa_gt=z,residue_index=torch.arange(4),chain_index=z,batch_index=z,mask=torch.ones(4,dtype=torch.bool),dssp_condition=torch.tensor([0,1,2,3]))
p={'pos':torch.zeros(4,5,3),'local':torch.zeros(4,2)}; u=torch.full((f.n_draws,4,4),.5)
a=f(d,p,uniform=u)[0]['pos'][:,0,0].clone(); d['dssp_condition']=torch.tensor([3,2,1,0]); b=f(d,p,uniform=u)[0]['pos'][:,0,0].clone(); eager=m.prepare_condition(d)['condition'][:,0]; f.invalidate_condition_cache(); c=f(d,p,uniform=u)[0]['pos'][:,0,0]
print('CACHE first',a.tolist(),'second',b.tolist(),'eager',eager.tolist(),'refreshed',c.tolist())
torch.manual_seed(7)
for length,count in [(8,4),(9,4),(4,1),(3,4)]:
    x=torch.randn(length,5,3); y=screw_replicate(x,count=count,angle=2*np.pi/count,radius=10,translation=0,chain_center=False)
    q=(length//count)*count
    print('SYM',length,count,'out_length',len(y),'identity',torch.equal(x,y),'remainder_equal',torch.equal(x[q:],y[q:]))
try: parse_sym('c0',replicate_count=None,screw_angle=None,screw_radius=0,screw_translation=0,sym_mean='False',sym_threshold=0)
except Exception as e: print('SYM c0',type(e).__name__,str(e))
mask=torch.tensor([False,False,True,True]); batch=torch.tensor([0,0,1,1]); w=base_weight(mask,batch)
print('LOSS weight',w.tolist(),'sum',w.sum().item())
off=torch.zeros(4,dtype=torch.bool)
print('LOSS all_masked_aa',aa_loss(torch.full((4,20),-3.),torch.zeros(4,dtype=torch.long),off,~off,off).item(),'all_masked_dssp',dssp_loss(torch.full((4,3),-1.),torch.zeros(4,dtype=torch.long),off,off).item())
for path in sorted(Path('fixtures').glob('*.npz')):
    with np.load(path) as x:
        print('FIXTURE',path.name,{k:str(x[k]) for k in ('jax_platform','jax_matmul_precision') if k in x.files})
print('FAST compute_distogram=True key_present', 'distogram' in f(d,p,uniform=u,compute_distogram=True)[0])
