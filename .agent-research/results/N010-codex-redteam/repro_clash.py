import torch
from salad_torch.inference.symmetry import clash_repulsion_grad
x=torch.tensor([[0.,0.,0.],[1.,2.,3.],[3.,-1.,2.],[0.1,0.,0.]])
g=clash_repulsion_grad(x)
y=x.clone().requires_grad_(True)
d=torch.sqrt((torch.maximum(y[:,None]-y[None,:],torch.tensor(1e-3))**2).sum(-1))
r=(d-2)/8
s=torch.clamp(1-r**6,min=1e-6)/torch.clamp(1-r**12,min=1e-6)
gref=torch.autograd.grad(s.mean(),y)[0]
print('CLASH torch',g.tolist())
print('CLASH upstream_formula',gref.tolist())
print('CLASH max_delta',float((g-gref).abs().max()))
from salad_torch.inference.symmetry import screw_replicate
p=torch.arange(4*5*3,dtype=torch.float32).reshape(4,5,3)
q=screw_replicate(p,count=1,angle=0.,radius=10.,translation=0.,chain_center=False)
upstream_c1=p-p[:,1,:].mean(0)
print('C1 input_CA_center',p[:,1,:].mean(0).tolist(),'torch_CA_center',q[:,1,:].mean(0).tolist(),'upstream_formula_CA_center',upstream_c1[:,1,:].mean(0).tolist(),'max_delta',float((q-upstream_c1).abs().max()))
