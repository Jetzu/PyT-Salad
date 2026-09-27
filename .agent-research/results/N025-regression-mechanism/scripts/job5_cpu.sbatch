#!/bin/bash -l
#SBATCH --job-name=n025_cpu
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=48
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_x/job5_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export OMP_NUM_THREADS=48
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
X=/hnvme/workspace/v181bb10-salad/n025_x
H=/hnvme/workspace/v181bb10-salad/n025
G=/hnvme/workspace/v181bb10-salad/n025_good
W=/tmp/n025cpu_$SLURM_JOB_ID; mkdir -p $W
IN=$W/inputs.pt
for L in 64 100; do
  echo "########## L=$L  GOOD (c5ab7a0)"
  $P $X/x_cpu_bits.py --root $G --num_aa $L --inputs $W/in_$L.pt --out $W/good_$L.npz 2>&1 | tail -4
  echo "########## L=$L  HEAD (fd9ba32), all defaults"
  $P $X/x_cpu_bits.py --root $H --num_aa $L --inputs $W/in_$L.pt --out $W/head_$L.npz 2>&1 | tail -4
  echo "########## L=$L  HEAD, SALAD_OPT_BATCHCOND=0 (P7 off)"
  SALAD_OPT_BATCHCOND=0 $P $X/x_cpu_bits.py --root $H --num_aa $L --inputs $W/in_$L.pt --out $W/head_nop7_$L.npz 2>&1 | tail -4
done
echo "########## COMPARE"
$P - <<'PY'
import glob, os, numpy as np
W = os.environ["W"] if "W" in os.environ else None
import sys
W = sorted(glob.glob("/tmp/n025cpu_*"))[-1]
for L in (64, 100):
    g = np.load(f"{W}/good_{L}.npz")
    for tag in ("head", "head_nop7"):
        p = f"{W}/{tag}_{L}.npz"
        if not os.path.exists(p):
            print(f"L={L} {tag}: MISSING"); continue
        h = np.load(p)
        print(f"--- L={L}  good vs {tag}")
        for k in g.files:
            a, b = g[k].astype(np.float64), h[k].astype(np.float64)
            if a.shape != b.shape:
                print(f"    {k:22s} SHAPE {a.shape} vs {b.shape}"); continue
            d = np.abs(a - b).max() if a.size else 0.0
            print(f"    {k:22s} max|d| = {d:.6e}   bitwise_equal={bool((g[k]==h[k]).all())}")
PY
rm -rf $W
echo DONE
