#!/bin/bash -l
#SBATCH --job-name=n028_fus
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor/logs/fus_%j.log
set -u
# PLACEMENT: 1x H100. Inductor codegen + a captured-graph profile; GPU-only by
# construction (the kernels under audit only exist on a CUDA compile). <1 GPU-h.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n028wt
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
# MANDATORY: without this every inductor kernel profiles as the literal `triton_`
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
cd $SALAD_TREE
nvidia-smi --query-gpu=name --format=csv,noheader
# one length per process, one arm per process
for arm in base o2; do
  if [ "$arm" = "o2" ]; then export SALAD_OPT_IPAMULSUM=1; else export SALAD_OPT_IPAMULSUM=0; fi
  D=/dev/shm/n028fus_${SLURM_JOB_ID}_$arm
  export TORCHINDUCTOR_CACHE_DIR=/dev/shm/n028c_${SLURM_JOB_ID}_$arm
  N028_OUT=$D $P -u $R/x_fusion.py
  mkdir -p $R/dump/$arm
  for f in output_code.py ir_pre_fusion.txt ir_post_fusion.txt fx_graph_readable.py fusion.log census.json; do
    [ -f $D/$f ] && gzip -c $D/$f > $R/dump/$arm/$f.gz
  done
  rm -rf $D $TORCHINDUCTOR_CACHE_DIR
  ls -la $R/dump/$arm
done
