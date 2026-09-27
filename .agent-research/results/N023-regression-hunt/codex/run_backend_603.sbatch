#!/bin/bash -l
#SBATCH --job-name=n023_back603
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:20:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/backend603_%j.log
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex
cd "$R/603"
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export TMPDIR=$R/tmp TORCHINDUCTOR_CACHE_DIR=$R/snapshot_cache TRITON_CACHE_DIR=$R/snapshot_triton
export SALAD_OPT_DETSEG=1 SALAD_OPT_AGGRFUSE=0
for bc in 0 1; do
    export SALAD_OPT_BATCHCOND=$bc TRACE_OUT=$R/backend603_trace_b${bc}.npz INPUT_OUT=$R/backend603_input_b${bc}.npz
    echo "### BACKEND603 BATCHCOND=$bc"
    "$P" -u "$R/run_backend_603.py" --fast --preset fast --num_aa 100 --num_designs 1 --seed 42 --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz --out_path "$R/backend603_out_b${bc}"
    echo "### EXIT=$?"
done
