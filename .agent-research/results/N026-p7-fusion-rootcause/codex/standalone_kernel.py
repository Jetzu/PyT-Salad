"""No salad imports. Exact Triton source + captured real kernel inputs."""
from pathlib import Path
import torch
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
D=Path(__file__).parent
saved=torch.load(D/'real_kernel_io.pt',map_location='cpu')
arg=saved['args']
inputs=[x.cuda() for x in arg[:16]]

def load(path,names):
    env={'async_compile':AsyncCompile()}
    exec(compile((D/path).read_text(),str(D/path),'exec'),env)
    env['async_compile'].wait(env)
    return [env[n] for n in names]

fused,=load('limit15_fused_kernel.py',['triton_poi_fused_add_clamp_eq_full_like_index_repeat_stack_sub_where_72'])
stack,rel=load('limit14_unfused_kernels.py',['triton_poi_fused_stack_72','triton_poi_fused_add_clamp_eq_full_like_index_repeat_sub_where_74'])
stream=get_raw_stream(0)
nb_f=torch.empty_like(saved['outputs'][0],device='cuda')
rel_f=torch.empty_like(saved['outputs'][1],device='cuda')
nb_u=torch.empty_like(nb_f); rel_u=torch.empty_like(rel_f)
fused.run(*inputs,nb_f,rel_f,28800,grid=grid(28800),stream=stream)
stack.run(*inputs[:13],nb_u,28800,grid=grid(28800),stream=stream)
rel.run(inputs[13],nb_u,inputs[14],inputs[15],rel_u,28800,grid=grid(28800),stream=stream)
torch.cuda.synchronize()
for title,a,b in [('fused vs captured nb',nb_f.cpu(),saved['outputs'][0]),('fused vs captured rel',rel_f.cpu(),saved['outputs'][1]),('fused vs unfused nb',nb_f,nb_u),('fused vs unfused rel',rel_f,rel_u)]:
    neq=(a!=b); maxabs=int((a-b).abs().max())
    print(title,'mismatch',int(neq.sum()),'/',neq.numel(),'max_abs',maxabs)
    if title=='fused vs unfused rel' and neq.any():
        i,j=neq.nonzero()[0].tolist()
        print('first rel mismatch',i,j,'raw_neighbour',int(nb_f[i,j]),'fused',int(a[i,j]),'unfused',int(b[i,j]))
