"""Compare P7 prologue eager and compiled on fixed real checkpoint tensors."""
import os
import sys
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
import torch
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision(os.environ.get('PROLOGUE_PRECISION', 'high'))
print('PRECISION', torch.get_float32_matmul_precision())
c = get_config('default_vp')
m = StructureDiffusionPredict(c).eval().cuda()
load_model(m, load_param_npz('/hnvme/workspace/v181bb10-salad/default_vp-200k.npz'), c, strict=True)
s = m.diffusion.diffusion_stack
print('SYNC', s.cond_pair_stack.in_sync(s.blocks))
L,A,B,K=100,5+c.augment_size,c.diffusion_depth,48
resi=torch.arange(L,device='cuda'); chain=torch.zeros_like(resi); batch=torch.zeros_like(resi)
mask=torch.ones(L,dtype=torch.bool,device='cuda')
g=torch.Generator(device='cuda').manual_seed(71)
nb=torch.randint(0,L,(B,L,K),generator=g,device='cuda')
for mode in ('zero','random'):
    pos=torch.zeros(L,A,3,device='cuda') if mode=='zero' else torch.randn(L,A,3,device='cuda',generator=g)
    args=(pos,None,nb,resi,chain,batch,mask)
    eager=s.cond_pair_stack(*args)
    compiled=torch.compile(s.cond_pair_stack.forward,mode='default',dynamic=False)(*args)
    torch.cuda.synchronize()
    pmax=float((eager[0]-compiled[0]).abs().max())
    peq=torch.equal(eager[0],compiled[0])
    seq=[]
    for bi,block in enumerate(s.blocks):
        seq.append(block.pair_2(pos,None,nb[bi],resi,chain,batch,mask)[0])
    seq=torch.stack(seq)
    smax=float((eager[0]-seq).abs().max())
    print('MODE',mode,'compiled_vs_eager',pmax,'equal',peq,'eager_vs_sequential',smax)
