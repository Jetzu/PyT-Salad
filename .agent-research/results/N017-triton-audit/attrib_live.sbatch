#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --job-name=n017_attriblive
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit/attriblive_%j.log
set -e
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n017live
export N017_OUT=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit/live
export N017_IDXMEAN=0
export TORCHINDUCTOR_CACHE_DIR=/tmp/n017l_ind_${SLURM_JOB_ID}
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=/tmp/n017l_dbg_${SLURM_JOB_ID}
cd $SALAD_TREE
python -u /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit/x_attrib.py
mkdir -p $N017_OUT/inductor
find /tmp/n017l_dbg_${SLURM_JOB_ID} \( -name 'output_code.py' -o -name 'fx_graph_readable.py' \) | while read fp; do
  d=$(basename $(dirname $fp)); mkdir -p $N017_OUT/inductor/$d; cp $fp $N017_OUT/inductor/$d/
done
du -sh $N017_OUT/inductor
