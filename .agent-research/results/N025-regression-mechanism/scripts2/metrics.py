import glob, importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location(
    'q', '/home/hpc/v181bb/v181bb10/SALAD_torch/tests/test_design_quality.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d, label = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(d + '/*.pdb'))
rows = [m.metrics(f) for f in files]
dv = [i for i, r in enumerate(rows) if r['diverged']]
ok = [r for r in rows if not r['diverged']]
cc = np.mean([r['ca_ca'] for r in ok]) if ok else float('nan')
rg = np.mean([r['rg'] for r in ok]) if ok else float('nan')
rgs = np.std([r['rg'] for r in ok]) if ok else float('nan')
print(f"ARMRESULT {label:22} n={len(rows):3} diverged={len(dv):3} "
      f"frac={len(dv)/max(len(rows),1):7.2%} ca_ca={cc:.4f} rg={rg:6.2f}+-{rgs:4.2f} "
      f"dividx={dv}", flush=True)
