#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --job-name=n017_attrib
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit/attrib_%j.log
set -e
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n017tree
export N017_OUT=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit
export TORCHINDUCTOR_CACHE_DIR=/tmp/n017b_ind_${SLURM_JOB_ID}
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=/tmp/n017b_dbg_${SLURM_JOB_ID}
cd $SALAD_TREE
python -u $N017_OUT/x_attrib.py
# keep only the small text artefacts; the debug dir also holds big caches
mkdir -p $N017_OUT/inductor
find /tmp/n017b_dbg_${SLURM_JOB_ID} -name 'output_code.py' -o -name 'fx_graph_readable.py' \
     -o -name 'ir_post_fusion.txt' -o -name 'ir_pre_fusion.txt' | while read fp; do
  d=$(basename $(dirname $fp)); mkdir -p $N017_OUT/inductor/$d; cp $fp $N017_OUT/inductor/$d/
done
du -sh $N017_OUT/inductor
ls -R $N017_OUT/inductor | head -40
