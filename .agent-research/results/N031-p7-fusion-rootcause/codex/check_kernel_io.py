"""Pure PyTorch unfused reference for the captured exact fused kernel outputs."""
import sys,torch
p=sys.argv[1]
d=torch.load(p,map_location='cpu')
a=d['args']; got_nb,got_rel=d['outputs']
print('kernel',d['kernel'],'arg_count',len(a),'nb_shape',tuple(got_nb.shape),'rel_shape',tuple(got_rel.shape))
for j in range(13,16):
 x=a[j]
 print('arg',j,'shape',tuple(x.shape),'minmax',int(x.min()),int(x.max()),'unique',int(x.unique().numel()))
# in_ptr15 is resi; in_ptr13/14 are batch/chain in either order.
resi=a[15]; chain=a[14]; batch=a[13]
n=100;b=6;k=48
nb=got_nb.reshape(b,n,k).long()
off=(torch.arange(b)*n)[:,None,None]
nb_t=torch.where(nb>=0,nb+off,off+n-1).reshape(b*n,k)
resi_t=resi.repeat(b)
chain_t=(chain[None]+off[:,:,0]).reshape(-1)
batch_t=(batch[None]+off[:,:,0]).reshape(-1)
same_chain=chain_t[:,None]==chain_t[nb_t]
same_batch=batch_t[:,None]==batch_t[nb_t]
rel=resi_t[:,None]-resi_t[nb_t]
expected=torch.where(same_batch,torch.where(same_chain,rel.clamp(-32,32)+32,65),65)
delta=(expected!=got_rel)
print('relpos mismatches',int(delta.sum()),'/',delta.numel(),'fraction',float(delta.float().mean()),'max_abs',int((expected-got_rel).abs().max()))
if delta.any():
 i,j=delta.nonzero()[0].tolist()
 print('first mismatch',i,j,'raw_nb',int(nb.reshape(n*b,k)[i,j]),'expected',int(expected[i,j]),'fused',int(got_rel[i,j]))
