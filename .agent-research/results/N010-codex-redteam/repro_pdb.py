import numpy as np
from salad_torch.io.pdb import Protein as TProtein,to_pdb as t_pdb
from salad.aflib.common.protein import Protein as JProtein,to_pdb as j_pdb
for chain in ([0],[0,0],[0,1],[0,2]):
 n=len(chain); p=np.arange(n*37*3,dtype=np.float32).reshape(n,37,3)/10; aa=np.arange(n,dtype=np.int32); mask=np.zeros((n,37),np.float32); mask[:,:4]=1
 fields=(p,aa,mask,np.arange(1,n+1),np.array(chain),np.zeros((n,37),np.float32))
 a=t_pdb(TProtein(*fields)); b=j_pdb(JProtein(*fields))
 print('PDB',chain,'identical',a==b,'lines',len(a.splitlines()))
