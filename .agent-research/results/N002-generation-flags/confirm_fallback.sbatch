#!/bin/bash -l
#SBATCH --job-name=salad_oa_fallback
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:05:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N002-generation-flags/logs/oa_fallback_%j.log
set -u
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
OUT=.agent-research/results/N002-generation-flags/out/oa_fallback_confirm
mkdir -p "$OUT"
# steps=60 seed=1 was one of the 15/15 combinations that reproducibly hit
# reorder_amino_acids's safety-net RuntimeError before the write_result_pdb
# fallback + residue_index fix (see check_order_agnostic.py / oa_sweep log).
# Confirm the fixed CLI: (a) exits 0, (b) prints the fallback warning,
# (c) writes a well-formed, uniquely-numbered 32-residue PDB.
$P scripts/generate.py --num_aa 32 --num_steps 60 --out_steps 50 \
  --order_agnostic --seed 1 --out_path "$OUT"
echo "exit code: $?"
$P -c "
import sys; sys.path.insert(0, '.')
from salad_torch.inference.template import read_pdb
p = read_pdb('$OUT/design_0.pdb')
print('n residues parsed:', p.aatype.shape[0])
print('residue_index unique count:', len(set(p.residue_index.tolist())))
print('residue_index:', p.residue_index)
"
