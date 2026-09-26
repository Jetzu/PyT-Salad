import numpy as np,torch
from tests.harness.check_loss import CONFIG,_t
from salad_torch.losses.total import total_loss
D=np.load('fixtures/loss_ref.npz'); orig={k[3:]:_t(D[k]) for k in D.files if k.startswith('in_')}; result={k[4:]:_t(D[k]) for k in D.files if k.startswith('res_')}
for n in (1,2,8):
    data={k:v[:n].clone() for k,v in orig.items()}
    r={k:(v[:,:n].clone() if k=='trajectory' else v[:n].clone()) for k,v in result.items()}
    data['batch_index'][:]=0; data['chain_index'][:]=0; data['residue_index']=torch.arange(n)
    r['distogram_neighbours'][:]=0
    for label in ('normal','masked'):
        if label=='masked': data['mask'][:]=False
        try:
            total,terms=total_loss(CONFIG,data,r,fape_uniform=torch.full((n,n),.5))
            print('EDGE',n,label,'total',float(total),'finite',bool(torch.isfinite(total)),'zero_terms',[k for k,v in terms.items() if float(v)==0],flush=True)
        except Exception as e: print('EDGE',n,label,type(e).__name__,str(e)[:300],flush=True)
n=8
data={k:v[:n].clone() for k,v in orig.items()}; r={k:(v[:,:n].clone() if k=='trajectory' else v[:n].clone()) for k,v in result.items()}
data['batch_index'][:]=0; data['chain_index'][:]=0; data['residue_index']=torch.arange(n); r['distogram_neighbours'][:]=0
one,t1=total_loss(CONFIG,data,r,fape_uniform=torch.full((n,n),.5))
two_data={k:torch.cat([v,v.clone()],0) for k,v in data.items()}; two_data['batch_index'][n:]=1; two_data['chain_index'][n:]=1
two_r={k:(torch.cat([v,v.clone()],1) if k=='trajectory' else torch.cat([v,v.clone()],0)) for k,v in r.items()}; two_r['distogram_neighbours'][n:]=n
two,t2=total_loss(CONFIG,two_data,two_r,fape_uniform=torch.full((2*n,2*n),.5))
print('BATCH_DUP one',float(one),'two',float(two),'delta',float(two-one))
print('BATCH_DUP term_delta',{k:round(float(t2[k]-t1[k]),6) for k in t1})
