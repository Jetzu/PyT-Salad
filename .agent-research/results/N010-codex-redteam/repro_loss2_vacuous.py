import importlib.util,numpy as np
from pathlib import Path
p=Path('.agent-research/results/N010-codex-redteam/loss_ref2_corrupt.npz')
with np.load('fixtures/loss_ref2.npz') as x: d={k:x[k] for k in x.files}
d['loss_pos']=d['loss_pos']+1000; np.savez(p,**d)
s=importlib.util.spec_from_file_location('loss2','tests/harness/check_loss2.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.FIX=str(p)
print('check_loss2 returned',m.main())
