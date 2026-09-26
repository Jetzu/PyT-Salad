#!/bin/bash
#SBATCH --job-name=n010-sym-template
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N010-codex-redteam/repro_sym_template_slurm.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export PYTHONPATH=.
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u scripts/generate.py --device cuda --num_steps 2 --out_steps 1 --num_designs 1 --sym c4 --screw_radius 10 --template '.agent-research/results/N010-codex-redteam/generated/design_0.pdb,FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF' --template_aa --out_path .agent-research/results/N010-codex-redteam/generated_sym_template
