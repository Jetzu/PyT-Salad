#!/bin/bash -l
#SBATCH --job-name=n021_surf
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:20:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/surf_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n021tree
# job-local inductor cache: shared across the processes of THIS job so only the
# first at a given (L, B) pays the compile, and gone with the job (/hnvme is
# inode-capped).
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
PRE=${SALAD_PRESET:?}
L=${SALAD_L:?}
B=${SALAD_B:?}
GS=${SALAD_GS:?}
echo "### node $(hostname) $(date) preset=$PRE L=$L B=$B G in {$GS}"
python -c "import torch;p=torch.cuda.get_device_properties(0);print('GPU',p.name,p.multi_processor_count,'SM',round(p.total_memory/2**30,1),'GiB')"
for G in $GS; do
  timeout 1400 python -u $R/x_surface.py --preset $PRE --length $L \
      --batch_size $B --group_size $G \
      2>&1 | grep -E "^CELL|Error|Traceback" || echo "CELL_TIMEOUT preset=$PRE L=$L B=$B G=$G"
done
echo "### done $(date)"
rm -rf $TORCHINDUCTOR_CACHE_DIR
