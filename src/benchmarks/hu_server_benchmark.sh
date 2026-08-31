#!/bin/bash

#SBATCH --job-name=dcegm_bench
#SBATCH --mail-user=maximilian.blesch@hu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --qos=standard

# Runs all three dcegm-engine benchmark specs (see SPEC_LIST in run_benchmark.py) in
# a single job, against whichever dcegm branch is currently checked out in
# submodules/dcegm. Round 1 (see output_v1_uniform_grid_branch_check/) already
# confirmed dcegm-branch choice doesn't change results, so unlike round 1 there's no
# branch switching here -- just make sure submodules/dcegm is on the branch you
# want before submitting. Submit from src/benchmarks/.
#
# Sizing --time: this runs 3 specs, each with an untimed warm-up + timed solve and
# simulate. Round 1's smaller/synthetic grid took ~50s/spec end to end; production's
# real grid (used throughout this round) is a similar size, so --time=1:00:00 should
# be generous, but check a real log before shrinking it.

module add cuda/12.4.1
# The compute node's bare shell doesn't have git on PATH by default (unlike the
# login node) -- run_benchmark.py needs it to record the dcegm branch/commit.
module add git/2.51.0

python run_benchmark.py
python make_table.py
