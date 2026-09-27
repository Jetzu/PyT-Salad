"""Adaptive first-bad-fusion search; one length and one saved state."""
import os, subprocess, sys, numpy as np
from pathlib import Path
D=Path(__file__).parent
ROOT=Path.cwd(); T=Path(os.environ['N026_TMP'])
PY=sys.executable
base=[PY,str(D/'bisect_forward.py'),'--root',str(ROOT),'--state',str(T/'state.pt'),'--p7','on']
def run(lim):
 out=T/f'lim{lim}.npz'
 log=D/f'lim{lim}.log'
 with log.open('w') as f:
  p=subprocess.run(base+['--out',str(out),'--compiled','--fusion-limit',str(lim)],stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:
  print('FAILED',lim,p.returncode,log,flush=True); raise SystemExit(1)
 ref=np.load(T/'eager.npz'); got=np.load(out)
 d=float(np.abs(ref['pos']-got['pos']).max())
 n=sum(1 for line in log.open() if line.startswith('[fusion]'))
 print('LIMIT',lim,'pos_abs',d,'candidate_calls',n,flush=True)
 return d
with (D/'dump_bisect.log').open('w') as f:
 p=subprocess.run(base+['--out',str(T/'unused.npz'),'--dump'],stdout=f,stderr=subprocess.STDOUT)
 if p.returncode: raise SystemExit(p.returncode)
with (D/'eager_bisect.log').open('w') as f:
 p=subprocess.run(base+['--out',str(T/'eager.npz')],stdout=f,stderr=subprocess.STDOUT)
 if p.returncode: raise SystemExit(p.returncode)
lo=0; hi=2000
if run(lo)>0.01: raise SystemExit('limit 0 is already wrong')
if run(hi)<0.01: raise SystemExit('limit 2000 is still correct')
while hi-lo>1:
 mid=(hi+lo)//2
 if run(mid)>0.01: hi=mid
 else: lo=mid
print('FIRST_BAD',hi,'LAST_GOOD',lo,flush=True)
