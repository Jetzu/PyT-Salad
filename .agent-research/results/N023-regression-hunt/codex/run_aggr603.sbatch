#!/bin/bash -l
#SBATCH --job-name=n023_ag603
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:25:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/ag603_%j.log
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex
cd "$R/603"
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export TMPDIR=$R/tmp TORCHINDUCTOR_CACHE_DIR=$R/snapshot_cache TRITON_CACHE_DIR=$R/snapshot_triton
export SALAD_OPT_DETSEG=1 SALAD_OPT_BATCHCOND=0
for af in 0 1; do
    export SALAD_OPT_AGGRFUSE=$af TRACE_OUT=$R/ag603_a${af}.npz
    echo "### AG603 BATCHCOND=0 AGGRFUSE=$af DETSEG=1"
    "$P" -u "$R/run_trace_603.py" --fast --preset fast --num_aa 100 --num_designs 1 --seed 42 --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz --out_path "$R/ag603_out_a${af}"
    echo "### EXIT=$?"
done
"$P" - <<'PY'
import numpy as np
r='/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex'
a=np.load(r+'/ag603_a0.npz')['ca']; b=np.load(r+'/ag603_a1.npz')['ca']
for i in [0,1,2,10,50,100,200]: print('STEP',i,'max_abs',float(np.max(np.abs(a[i]-b[i]))))
PY
