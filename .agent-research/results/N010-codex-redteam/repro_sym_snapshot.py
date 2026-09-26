import math,torch
from salad_torch.diffusion.sampling import make_initial_data,sample,SamplerConfig
from salad_torch.inference.symmetry import screw_replicate,rot_y
from salad_torch.io.pdb import snapshot_to_protein
L=8; ids=torch.arange(L); z=torch.zeros(L,dtype=torch.long)
data,prev=make_initial_data(L,ids,z,None,device='cpu')
x=torch.randn(L,20,3,generator=torch.Generator().manual_seed(19))
def step(d,p,generator=None):
    return dict(pos=x,atom_pos=x[:,:14].clone(),aa=torch.zeros(L,20),aatype=z),p
sn=sample(step,data,prev,SamplerConfig(num_steps=1,out_steps=(0,)),post_step_fn=lambda p,t:screw_replicate(p,count=4,angle=math.pi/2,radius=10,translation=0,chain_center=False))[0]
R=rot_y(math.pi/2)
def residual(ca):
    n=L//4
    return max(float((ca[i*n:(i+1)*n]-ca[:n]@torch.linalg.matrix_power(R,i)).abs().max()) for i in range(1,4))
protein=snapshot_to_protein(dict(sn,aa_logits=torch.zeros(L,20)))
print('SNAPSHOT pos_C4_maxcoord',residual(sn['pos'][:,1]))
print('SNAPSHOT atom_pos_C4_maxcoord',residual(sn['atom_pos'][:,1]))
print('WRITTEN atom37_CA_C4_maxcoord',residual(torch.from_numpy(protein.atom_positions[:,1])))
