"""No salad imports. Reproducer for Inductor stack + indirect gather fusion."""
import os,torch
import torch._inductor.config as config
if os.getenv('NOFUSE')=='1': config.max_fusion_size=1
N=100; B=6; K=48
G=torch.Generator().manual_seed(20260927)
fixed=torch.randint(0,N,(N,32),generator=G).cuda()
indices=[]; scores=[]
for _ in range(B):
 idx=torch.randint(0,N,(N,N),generator=G,dtype=torch.int16)
 score=torch.rand((N,N),generator=G)
 score[torch.rand((N,N),generator=G)<0.1]=float('inf')
 indices.append(idx.cuda()); scores.append(score.cuda())
resi=torch.arange(N,device='cuda'); chain=torch.zeros_like(resi); batch=chain.clone()
def fn(fixed,i0,s0,i1,s1,i2,s2,i3,s3,i4,s4,i5,s5,resi,chain,batch):
    items=[]
    for ix,sc in ((i0,s0),(i1,s1),(i2,s2),(i3,s3),(i4,s4),(i5,s5)):
        tail=ix[:,:16].long()
        picked=sc.gather(1,tail)
        tail=torch.where(picked<float('inf'),tail,-1)
        items.append(torch.cat((fixed,tail),dim=1))
    neighbours=torch.stack(items)
    off=(torch.arange(6,device=neighbours.device)*100)[:,None,None]
    nb=torch.where(neighbours>=0,neighbours+off,off+99).reshape(600,48)
    resi_t=resi.repeat(6)
    chain_t=(chain[None]+off[:,:,0]).reshape(-1)
    batch_t=(batch[None]+off[:,:,0]).reshape(-1)
    same_chain=chain_t[:,None]==chain_t[nb]
    same_batch=batch_t[:,None]==batch_t[nb]
    rel=resi_t[:,None]-resi_t[nb]
    binned=rel.clamp(-32,32)+32
    binned=torch.where(same_chain,binned,65)
    binned=torch.where(same_batch,binned,65)
    return binned
args=(fixed,*sum(([i,s] for i,s in zip(indices,scores)),[]),resi,chain,batch)
with torch.no_grad():
 eager=fn(*args)
 compiled=torch.compile(fn,mode='default',dynamic=False)(*args)
wrong=(eager!=compiled)
print('torch',torch.__version__,'max_fusion_size',config.max_fusion_size,'mismatches',int(wrong.sum()),'of',wrong.numel(),'max_abs',int((eager-compiled).abs().max()))
if wrong.any():
 x,y=wrong.nonzero()[0].tolist()
 print('first',x,y,'eager',int(eager[x,y]),'compiled',int(compiled[x,y]))
