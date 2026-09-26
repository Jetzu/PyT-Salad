import os,sys,time
import numpy as np, torch
sys.path.insert(0,os.getcwd())
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model,load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
D=np.load('fixtures/model_default_vp.npz')
def t(a):
    return torch.from_numpy(a.astype(np.int64 if np.issubdtype(a.dtype,np.integer) else np.float32)).cuda() if a.dtype!=np.bool_ else torch.from_numpy(a).cuda()
data={k[3:]:t(D[k]) for k in D.files if k.startswith('in_') and k not in ('in_prev_pos','in_prev_local')}
for k in ('pos_noised','seq_noised','t_pos','t_seq'): data[k]=t(D['noised_'+k])
prev={'pos':t(D['in_prev_pos']),'local':t(D['in_prev_local'])}; u=t(D['fixed_uniform']); L=len(data['pos']); c=get_config('default_vp')
m=StructureDiffusionPredict(c).eval().cuda(); load_model(m,load_param_npz('/hnvme/workspace/v181bb10-salad/default_vp-200k.npz'),c,strict=True)
with torch.no_grad():
    us=u.unsqueeze(0).expand(1+3*c.diffusion_depth,-1,-1).contiguous()
    print('uniform_shapes',tuple(u.shape),tuple(us.shape),flush=True)
    start=time.time(); f=FastDenoiser(m,L,device='cuda').warmup(); print('warmup_s',time.time()-start,flush=True)
    a=f(data,prev,uniform=us)[0]; ea=m(data,prev,uniform=u)[0]
    print('base_max',float((a['pos']-ea['pos']).abs().max()),flush=True)
    hold=a['pos'].clone(); orig=a['pos']
    b=f(data,prev,uniform=us)[0]
    print('repeat_max',float((a['pos']-b['pos']).abs().max()),'held_change',float((orig-hold).abs().max()),flush=True)
    d2=dict(data); d2['dssp_condition']=torch.tensor(([0,1]* (L//2))[:L],device='cuda')
    d3=dict(data); d3['dssp_condition']=d2['dssp_condition'].flip(0).clone()
    x=f(d2,prev,uniform=us)[0]; y=f(d3,prev,uniform=us)[0]; ey=m(d3,prev,uniform=u)[0]
    print('collision_sums',int(d2['dssp_condition'].sum()),int(d3['dssp_condition'].sum()),'fast_cross',float((x['pos']-y['pos']).abs().max()),'fast_vs_eager',float((y['pos']-ey['pos']).abs().max()),flush=True)
    f.invalidate_condition_cache(); z=f(d3,prev,uniform=us)[0]
    print('refresh_vs_eager',float((z['pos']-ey['pos']).abs().max()),flush=True)
    f2=FastDenoiser(m,L,device='cuda').warmup(); q=f2(d3,prev,uniform=us)[0]; r=f(d2,prev,uniform=us)[0]; s=f2(d3,prev,uniform=us)[0]
    print('interleave_f2_change',float((q['pos']-s['pos']).abs().max()),flush=True)
    dlong={k:(torch.cat([v,v[:1]],0) if torch.is_tensor(v) and v.ndim>0 and v.shape[0]==L and not (v.ndim>1 and v.shape[1]==L) else v) for k,v in d3.items()}
    for k in ('dmap','dmap_mask','omap','chain_contacts'):
        if k in dlong and torch.is_tensor(dlong[k]) and dlong[k].shape[:2]==(L,L):
            v=dlong[k]; dlong[k]=torch.nn.functional.pad(v,(0,0,0,1,0,1)) if v.ndim==3 else torch.nn.functional.pad(v,(0,1,0,1))
    plong={k:torch.cat([v,v[:1]],0) for k,v in prev.items()}
    try:
        o=f(dlong,plong,uniform=torch.full((f.n_draws,L+1,L+1),.5,device='cuda'))[0]
        print('length_mismatch_returned',len(o['pos']),flush=True)
    except Exception as e: print('length_mismatch_error',type(e).__name__,str(e)[:300],flush=True)
