#!/bin/bash -l
#SBATCH --job-name=salad_e2e
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N009-speed-proposals/logs/e2e_%j.log
# End-to-end regression: does any proposal change the WRITTEN STRUCTURE?
# Same seed, same length, same steps; the PDBs must be byte-identical.
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
O=.agent-research/results/N009-speed-proposals/e2e
rm -rf $O; mkdir -p $O
gen () { tag=$1; shift; echo "=== $tag  ($*)"; \
  env "$@" $P -u scripts/generate.py --num_aa 100 --num_designs 3 --fast \
      --seed 1234 --out_path $O/$tag; }
# the shipping configuration
gen allon  SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=1 SALAD_OPT_LEAN=1
# P2 off, everything else on  -> isolates "does skipping the heads change output"
gen nolean SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=1 SALAD_OPT_LEAN=0
# every proposal off -> the pre-N009 reference trajectory
gen alloff SALAD_OPT_GATHER=0 SALAD_OPT_HOIST=0 SALAD_OPT_NOPC=0 SALAD_OPT_LEAN=0
# reproducibility control: same flags as allon, second run
gen allon2 SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=1 SALAD_OPT_LEAN=1
echo "=== sha256 of every written PDB"
for t in allon nolean alloff allon2; do
  for f in $O/$t/design_*.pdb; do echo "$(sha256sum $f | cut -c1-16)  $f"; done
done
echo "=== byte comparison against allon"
for t in nolean alloff allon2; do
  for i in 0 1 2; do
    if cmp -s $O/allon/design_$i.pdb $O/$t/design_$i.pdb; then r=IDENTICAL; else r=DIFFERS; fi
    echo "  design_$i  allon vs $t : $r"
  done
done
