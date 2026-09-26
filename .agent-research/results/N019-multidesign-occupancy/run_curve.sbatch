#!/bin/bash -l
#SBATCH --job-name=n019_curve
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N019-multidesign-occupancy/logs/curve_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N019-multidesign-occupancy
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n019tree
# job-local inductor cache: shared across the N processes of THIS job so only
# the first pays the compile, and gone with the job (/hnvme is inode-capped).
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
PRE=${SALAD_PRESET:?}
L=${SALAD_L:?}
echo "### node $(hostname) $(date) preset=$PRE L=$L"
python -c "import torch;p=torch.cuda.get_device_properties(0);print('GPU',p.name,p.multi_processor_count,'SM',round(p.total_memory/2**30,1),'GiB')"
# ONE (preset, L, N) per PROCESS - several graph captures in one process is
# the same mistake as several shapes in one process (N009/N016).
for N in 1 2 3 4 6 8 12 16; do
  timeout 900 python -u $R/x_curve.py --preset $PRE --length $L --group_size $N \
      2>&1 | grep -E "^CURVE|Error|OutOfMemory|Traceback" || echo "CURVE_FAIL preset=$PRE L=$L N=$N"
done
timeout 900 python -u $R/x_curve.py --preset $PRE --length $L --group_size auto \
    2>&1 | grep -E "^CURVE|Error|Traceback" || echo "CURVE_FAIL preset=$PRE L=$L N=auto"
echo "### done $(date)"
rm -rf $TORCHINDUCTOR_CACHE_DIR
