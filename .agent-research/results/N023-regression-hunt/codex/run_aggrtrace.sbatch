#!/bin/bash -l
#SBATCH --job-name=n023_agtrace
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:25:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/agtrace_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=$PWD/.agent-research/results/N023-regression-hunt/codex
export TMPDIR=$R/tmp TORCHINDUCTOR_CACHE_DIR=$R/torch_cache TRITON_CACHE_DIR=$R/triton_cache
export SALAD_OPT_DETSEG=1 SALAD_OPT_BATCHCOND=0
for af in 0 1; do
    export SALAD_OPT_AGGRFUSE=$af TRACE_OUT=$R/agtrace_a${af}.npz
    echo "### AGTRACE BATCHCOND=0 AGGRFUSE=$af DETSEG=1"
    "$P" -u "$R/run_trace.py" --fast --preset fast --num_aa 100 --num_designs 1 --seed 42 --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz --out_path "$R/agtrace_out_a${af}"
    echo "### EXIT=$?"
done
"$P" - <<'PY'
import numpy as np
r='.agent-research/results/N023-regression-hunt/codex'
a=np.load(r+'/agtrace_a0.npz')['ca']; b=np.load(r+'/agtrace_a1.npz')['ca']; c=np.load(r+'/trace_b0.npz')['ca']
for label,x,y in [('repeat_a0',a,c),('a0_vs_a1',a,b)]:
 print('PAIR',label)
 for i in [0,1,2,10,50,100,200]:
  print('STEP',i,'max_abs',float(np.max(np.abs(x[i]-y[i]))))
PY
