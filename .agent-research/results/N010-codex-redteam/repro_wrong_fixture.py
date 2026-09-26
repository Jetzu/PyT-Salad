import importlib.util,numpy as np
from pathlib import Path
p=Path('.agent-research/results/N010-codex-redteam/model_wrong_but_accepted.npz')
with np.load('fixtures/model_default_vp.npz') as x: d={k:x[k] for k in x.files}
d['jax_platform']=np.asarray('gpu'); d['jax_matmul_precision']=np.asarray('highest')
d['out_raw_angles']=d['out_raw_angles']+np.float32(.01)
np.savez(p,**d)
s=importlib.util.spec_from_file_location('model_test','tests/test_model_parity.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.FIX=str(p)
print('INJECT out_raw_angles +0.01; provenance falsely gpu/highest')
print('test_return',m.main())
