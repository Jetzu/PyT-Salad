import numpy as np
for path in ('.agent-research/results/N010-codex-redteam/generated_sym_template/design_0.pdb',):
 lines=open(path).read().splitlines()
 ca=np.array([[float(s[i:i+8]) for i in (30,38,46)] for s in lines if s.startswith('ATOM') and s[12:16].strip()=='CA'])
 angle=np.pi/2; R=np.array([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])
 n=len(ca)//4
 delta=np.concatenate([ca[i*n:(i+1)*n]-ca[:n]@np.linalg.matrix_power(R,i) for i in range(1,4)])
 print('PDB_C4',path,'ca_count',len(ca),'rms',float(np.sqrt(np.mean(np.sum(delta**2,axis=-1)))),'max',float(np.max(np.linalg.norm(delta,axis=-1))))
