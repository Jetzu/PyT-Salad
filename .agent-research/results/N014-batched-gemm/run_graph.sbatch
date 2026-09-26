#!/bin/bash -l
#SBATCH --job-name=n014_graph
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:35:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/graph_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
G=.agent-research/results/N012-codex-floor-audit/graph_audit.py
export TORCHINDUCTOR_CACHE_DIR=/tmp/n014_ind_$SLURM_JOB_ID
md5sum salad_torch/modules/cond_prologue.py salad_torch/modules/diffusion.py
for C in default_vp default_vp_scaled default_ve_scaled multimotif_vp default_vp_cyclic; do
  echo "#### BATCHCOND=0 $C ####"
                        $P -u $G --config $C || true
  echo "#### BATCHCOND=1 $C ####"
  SALAD_OPT_BATCHCOND=1 $P -u $G --config $C || true
done
