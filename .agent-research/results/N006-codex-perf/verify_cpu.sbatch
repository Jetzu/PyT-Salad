#!/bin/bash -l
#SBATCH --job-name=salad_n006_parity
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N006-codex-perf/parity_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N006-codex-perf/codex
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
pids=()
for C in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
    SALAD_CONFIG=$C $P tests/test_model_parity.py > ../parity_model_${C}.log 2>&1 &
    pids+=("$!")
done
$P tests/test_geometry_parity.py > ../parity_geometry.log 2>&1 & pids+=("$!")
$P tests/test_stack_parity.py > ../parity_stack.log 2>&1 & pids+=("$!")
$P tests/test_encoder_parity.py > ../parity_encoder.log 2>&1 & pids+=("$!")
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for f in ../parity_model_*.log ../parity_geometry.log ../parity_stack.log ../parity_encoder.log; do
    echo "=== $f ==="
    tail -5 "$f"
done
exit "$status"
