#!/bin/bash -l
#SBATCH --job-name=n023_trace
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:35:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/trace_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=$PWD/.agent-research/results/N023-regression-hunt/codex
export TMPDIR=$R/tmp TORCHINDUCTOR_CACHE_DIR=$R/torch_cache TRITON_CACHE_DIR=$R/triton_cache
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export SALAD_OPT_DETSEG=1 SALAD_OPT_AGGRFUSE=0
for bc in 0 1; do
    export SALAD_OPT_BATCHCOND=$bc TRACE_OUT=$R/trace_b${bc}.npz
    echo "### TRACE BATCHCOND=$bc AGGRFUSE=0 DETSEG=1"
    "$P" -u "$R/run_trace.py" --fast --preset fast --num_aa 100 --num_designs 1 --seed 42 --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz --out_path "$R/trace_out_b${bc}"
    echo "### EXIT=$?"
done
"$P" - <<'PY'
import numpy as np
r='.agent-research/results/N023-regression-hunt/codex'
a=np.load(r+'/trace_b0.npz')['ca']
b=np.load(r+'/trace_b1.npz')['ca']
for step in [0,1,2,3,4,5,10,20,40,50,75,100,125,150,175,200]:
    x,y=a[step],b[step]
    diff=float(np.max(np.abs(x-y)))
    rms=float(np.sqrt(np.mean((x-y)**2)))
    rga=float(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1))))
    rgb=float(np.sqrt(np.mean(np.sum((y-y.mean(0))**2,axis=1))))
    print('STEP',step,'max_abs',f'{diff:.6g}','rms',f'{rms:.6g}','rg_off',f'{rga:.3f}','rg_on',f'{rgb:.3f}')
PY
