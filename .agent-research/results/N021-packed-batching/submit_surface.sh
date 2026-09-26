#!/bin/bash
# One job per (L, batch_size); the group sizes sweep inside it as separate
# PROCESSES (one capture per process, N019's rule), sharing the job-local
# inductor cache so only the first pays the compile for that shape.
set -u
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching
PRE=${1:-fast}
submit () { # L B "G list"
  sbatch --job-name=n021s_${2}x${1} \
     --export=ALL,SALAD_PRESET=$PRE,SALAD_L=$1,SALAD_B=$2,SALAD_GS="$3" \
     $R/run_surface.sbatch
}
submit 50   1 "1 2 4 8 16"
submit 50   2 "1 2 4 8"
submit 50   4 "1 2 4 8"
submit 50   8 "1 2 4"
submit 100  1 "1 2 4 8 16"
submit 100  2 "1 2 4 8"
submit 100  4 "1 2 4"
submit 100  8 "1 2"
submit 200  1 "1 2 4 8"
submit 200  2 "1 2 4"
submit 200  4 "1 2"
submit 200  8 "1"
submit 400  1 "1 2 4 8"
submit 400  2 "1 2 4"
submit 400  4 "1 2"
submit 1000 1 "1 2 4"
submit 1000 2 "1 2"
submit 1000 4 "1"
