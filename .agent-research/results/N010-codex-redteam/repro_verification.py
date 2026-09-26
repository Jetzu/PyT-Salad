import importlib.util, numpy as np
from pathlib import Path
out=Path('.agent-research/results/N010-codex-redteam')
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
src='fixtures/loss_ref.npz'; dst=out/'loss_ref_corrupt.npz'
with np.load(src) as x: d={k:x[k] for k in x.files}
d['loss_pos']=d['loss_pos']+1000
np.savez(dst,**d)
m=load('tests/harness/check_loss.py','loss_check'); m.FIX=str(dst)
print('LOSS CHECK corrupted pos reference by +1000',flush=True)
print('LOSS CHECK main returns',m.main(),flush=True)
src='fixtures/model_default_vp.npz'; dst=out/'model_fake_provenance.npz'
with np.load(src) as x: d={k:x[k] for k in x.files}
d['jax_platform']=np.asarray('gpu'); d['jax_matmul_precision']=np.asarray('highest')
np.savez(dst,**d)
m=load('tests/test_model_parity.py','model_test'); m.FIX=str(dst)
print('MODEL TEST spoofed GPU backend on original CPU fixture',flush=True)
print('MODEL TEST main returns',m.main(),flush=True)
