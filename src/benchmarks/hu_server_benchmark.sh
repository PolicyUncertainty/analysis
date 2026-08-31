#!/bin/bash

#SBATCH --job-name=dcegm_bench
#SBATCH --mail-user=maximilian.blesch@hu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:30:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --qos=standard

# Runs all four dcegm-engine benchmark cases (see run_benchmark.py) in a single job,
# switching the dcegm submodule branch between cases. Submit from src/benchmarks/.
#
# Sizing --time: this bundles 4 solve+simulate runs into one job instead of 4
# separate submissions. Size it well above 4x a single case's runtime (the SPECS
# grid sizes in run_benchmark.py determine that; check a single --case run's log
# first if unsure). If one case fails partway, the others still write their own
# results/<case_name>.pkl independently, so a partial job still leaves usable
# output -- rerun just the missing case with --case.

module add cuda/12.4.1
# The compute node's bare shell doesn't have git on PATH by default (unlike the
# login node) -- this script needs it for the branch checkouts below.
module add git/2.51.0

DCEGM_DIR="../../submodules/dcegm"

if [ -n "$(git -C "$DCEGM_DIR" status --porcelain)" ]; then
    echo "ERROR: submodules/dcegm has uncommitted changes. Commit or stash them" \
         "before running this script -- otherwise they'd silently carry across" \
         "the branch checkouts below and contaminate every case after the first."
    exit 1
fi

run_case () {
    local case_name=$1
    local branch=$2
    echo "=== $case_name (dcegm branch: $branch) ==="
    if ! git -C "$DCEGM_DIR" checkout "$branch"; then
        echo "FATAL: could not check out dcegm branch '$branch', aborting the" \
             "whole job -- every later case would otherwise silently run on the" \
             "wrong branch."
        exit 1
    fi
    if ! python run_benchmark.py --case "$case_name"; then
        echo "WARNING: case '$case_name' failed -- continuing with the remaining" \
             "cases; this one just won't have a results/${case_name}.pkl."
    fi
}

run_case main_fues main
run_case main_dj   main
run_case new_dj    remove_endog
run_case new_fues  remove_endog

python make_table.py
