"""No salad imports: test fused stack -> tiled index -> relative-position bins."""
import argparse
import torch
import torch._inductor.config as config

p=argparse.ArgumentParser()
p.add_argument('--real',default='')
p.add_argument('--nofuse',action='store_true')
p.add_argument('--trace-dir',default='')
a=p.parse_args()
if a.nofuse: config.max_fusion_size=1
if a.trace_dir:
 config.trace.enabled=True
 config.trace.debug_dir=a.trace_dir
N,K,B=100,48,6
if a.real:
 d=torch.load(a.real,map_location='cpu')
 nbr=[x.cuda() for x in d['nbr']]
 resi,chain,batch=(d[k].cuda() for k in ('resi','chain','batch'))
else:
 g=torch.Generator().manual_seed(42)
 nbr=[torch.randint(-1,N,(N,K),generator=g,device='cpu').cuda() for _ in range(B)]
 resi=torch.arange(N,device='cuda')
 chain=torch.zeros(N,dtype=torch.int64,device='cuda')
 batch=chain.clone()
def fn(n0,n1,n2,n3,n4,n5,resi,chain,batch):
 neighbours=torch.stack([n0,n1,n2,n3,n4,n5])
 off=(torch.arange(6,device=neighbours.device)*100)[:,None,None]
 nb=torch.where(neighbours>=0,neighbours+off,off+99).reshape(600,48)
 resi_t=resi.repeat(6)
 chain_t=(chain[None]+off[:,:,0]).reshape(-1)
 batch_t=(batch[None]+off[:,:,0]).reshape(-1)
 same_chain=chain_t[:,None]==chain_t[nb]
 same_batch=batch_t[:,None]==batch_t[nb]
 rel=resi_t[:,None]-resi_t[nb]
 binned=rel.clamp(-32,32)+32
 other=torch.full_like(binned,65)
 binned=torch.where(same_chain,binned,other)
 binned=torch.where(same_batch,binned,other)
 return binned
args=(*nbr,resi,chain,batch)
with torch.no_grad():
 eager=fn(*args)
 compiled=torch.compile(fn,mode='default',dynamic=False)(*args)
d=(eager!=compiled)
print('torch',torch.__version__,'max_fusion_size',config.max_fusion_size,'real',bool(a.real))
print('mismatches',int(d.sum()),'of',d.numel(),'max_abs',int((eager-compiled).abs().max()))
if d.any():
 idx=d.nonzero()[0].tolist()
 print('first_idx',idx,'eager',int(eager[tuple(idx)]),'compiled',int(compiled[tuple(idx)]))
