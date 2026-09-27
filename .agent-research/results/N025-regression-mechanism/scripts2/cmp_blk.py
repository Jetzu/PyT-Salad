import sys
import numpy as np
A, B, label = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
ta, tb = A["trajectory"], B["trajectory"]
per = [float(np.abs(ta[i] - tb[i]).max()) for i in range(ta.shape[0])]
first = next((i for i, d in enumerate(per) if d > 1e-3), -1)
print(f"BLK {label:36} pos={np.abs(A['pos']-B['pos']).max():.3e} "
      f"first_block>1e-3={first:>2}  per-block=["
      + " ".join(f"{d:.2e}" for d in per) + "]", flush=True)
