import tarfile,re
from pathlib import Path
D=Path(__file__).parent
for arm in ('off','on'):
 with tarfile.open(D/f'trace_{arm}.tar.gz') as t:
  m=next(m for m in t.getmembers() if m.name.endswith('model__1_inference_1.1/output_code.py'))
  s=t.extractfile(m).read().decode()
 rows=[]
 for part in s.split('# Topologically Sorted Source Nodes: ')[1:]:
  first=part.split('\n',1)[0]
  match=re.search(r'@triton.jit\ndef (triton_\w+)\(',part)
  if match:
   src,aten=first.split(', Original ATen: ',1)
   rows.append((len(rows),match.group(1),src,aten))
 (D/f'{arm}.kernels.tsv').write_text('\n'.join(f'{i}\t{name}\t{src}\t{aten}' for i,name,src,aten in rows)+'\n')
 print(arm,len(rows),'kernels')
 for i,name,src,aten in rows[:80]:
  print(f'{i:3} {name[:70]:70} {src[:100]}')
