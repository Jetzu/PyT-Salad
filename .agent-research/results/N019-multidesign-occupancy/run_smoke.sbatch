#!/bin/bash -l
#SBATCH --job-name=n019_smoke
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N019-multidesign-occupancy/logs/smoke_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N019-multidesign-occupancy
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n019tree
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
W=${TMPDIR:-/dev/shm}/n019-$SLURM_JOB_ID
mkdir -p $W
echo "### node $(hostname) $(date)"
python -c "import torch;p=torch.cuda.get_device_properties(0);print('GPU',p.name,p.multi_processor_count,'SM',round(p.total_memory/2**30,1),'GiB','torch',torch.__version__)"

echo "########## A. curve smoke, exact L=100, N=1"
timeout 1200 $P -u $R/x_curve.py --preset exact --length 100 --group_size 1 --reps 40 --host_steps 30 2>&1 | tail -30
echo "########## B. curve smoke, exact L=100, N=4"
timeout 1200 $P -u $R/x_curve.py --preset exact --length 100 --group_size 4 --reps 40 --host_steps 30 2>&1 | tail -30
echo "########## C. curve smoke, exact L=100, N=auto"
timeout 1200 $P -u $R/x_curve.py --preset exact --length 100 --group_size auto --reps 40 --host_steps 30 2>&1 | tail -30

echo "########## D. identity smoke (short schedule), group 1 vs 4, DETSEG=1"
export SALAD_OPT_DETSEG=1
timeout 1800 $P -u $R/x_identity.py --group_size 1 --num_designs 4 --num_aa 100 --num_steps 40 --out_steps 30 --out $W/det_g1 2>&1 | tail -20
timeout 1800 $P -u $R/x_identity.py --group_size 4 --num_designs 4 --num_aa 100 --num_steps 40 --out_steps 30 --out $W/det_g4 2>&1 | tail -20
$P -u $R/cmp_identity.py $W/det_g1 $W/det_g4 DETSEG_g1_vs_g4 2>&1 | tail -20
unset SALAD_OPT_DETSEG
echo "### done $(date)"
rm -rf $W $TORCHINDUCTOR_CACHE_DIR
