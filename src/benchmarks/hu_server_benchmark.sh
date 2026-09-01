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

# Runs the druedahl_jorgensen assets_begin_of_period grid sweep: check_dj_grid.py
# (accuracy vs. the fues reference) and run_dj_grid_timing.py (solve time/memory for
# the same candidate set, see dj_candidates.py). No dcegm-branch switching. Submit
# from src/benchmarks/.
#
# Sizing --time: each script solves a fues reference plus one dj solve per candidate
# (10 candidates); check_dj_grid.py additionally simulates every one of those. A
# single solve+simulate on production-sized grids takes roughly 50s, so ~11x that
# for check_dj_grid.py (~9 min) plus ~11 solves for run_dj_grid_timing.py (~5 min)
# comfortably fits --time=1:00:00, but check a real log before shrinking it.

module add cuda/12.4.1

python check_dj_grid.py
python run_dj_grid_timing.py
