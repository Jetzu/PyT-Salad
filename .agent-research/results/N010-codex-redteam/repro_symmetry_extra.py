import torch,math
from salad_torch.inference.symmetry import screw_replicate,rot_y
x=torch.randn(8,5,3,generator=torch.Generator().manual_seed(8))
y=screw_replicate(x,count=4,angle=math.pi/2,radius=10,translation=0,chain_center=False)
r=rot_y(math.pi/2)
print('C4 residual',max(float((y[2*i:2*i+2]-y[:2]@torch.linalg.matrix_power(r,i)).abs().max()) for i in range(4)))
for count in (1,9,64):
 z=screw_replicate(x,count=count,angle=2*math.pi/count,radius=0,translation=0,chain_center=False)
 print('ORDER',count,'L',len(x),'unchanged',torch.equal(x,z))
