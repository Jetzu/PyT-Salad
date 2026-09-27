#!/bin/bash -l
#SBATCH --job-name=n023_inputs
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:20:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/inputs_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=$PWD/.agent-research/results/N023-regression-hunt/codex
export TMPDIR=$R/tmp TORCHINDUCTOR_CACHE_DIR=$R/torch_cache TRITON_CACHE_DIR=$R/triton_cache
export SALAD_OPT_DETSEG=1 SALAD_OPT_AGGRFUSE=0
for bc in 0 1; do
    export SALAD_OPT_BATCHCOND=$bc TRACE_OUT=$R/trace_inputs_b${bc}.npz INPUT_OUT=$R/inputs_b${bc}.npz
    echo "### INPUTS BATCHCOND=$bc"
    "$P" -u "$R/run_trace_inputs.py" --fast --preset fast --num_aa 100 --num_designs 1 --seed 42 --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz --out_path "$R/inputs_out_b${bc}"
    echo "### EXIT=$?"
done
"$P" - <<'PY'
import numpy as np
r='.agent-research/results/N023-regression-hunt/codex'
a=np.load(r+'/inputs_b0.npz'); b=np.load(r+'/inputs_b1.npz')
for k in a.files:
    x,y=a[k],b[k]
    print('INPUT',k,'equal',np.array_equal(x,y),'max_abs',float(np.max(np.abs(x.astype('float64')-y.astype('float64')))))
PY
