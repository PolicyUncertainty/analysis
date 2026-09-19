#!/bin/bash

#SBATCH --job-name=dcegm_bench
#SBATCH --mail-user=maximilian.blesch@hu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=120GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --qos=standard

# Runs run_infrastructure_benchmark.py: build, solve, likelihood and solve-and-simulate,
# each timed over its compile call and four steady-state calls, with device and host
# memory. Submit from src/benchmarks/.
#
# What to set before submitting, at the top of run_infrastructure_benchmark.py:
#   RUN_NAME    identifies this machine in the results (keep it stable per device)
#   SOLVE_MODE  "pooled" | "chunked_seq" | "chunked_parallel"
#   SMALL_RAM   True blocks the income-shock draws (income_shock_batch_size=1)
# and, here, the --gres line to the device you are measuring. RUN_NAME is only a
# label: it does not select a device, so change both together or the results will be
# filed under the wrong name.
#
# One run covers one (SOLVE_MODE, SMALL_RAM) pair and writes its own CSV, so a sweep
# is several submissions; the filenames do not collide.
#
# Sizing --time: the four stages each pay a compile once and then run four more
# times. A production-sized solve is roughly a minute once compiled, compilation is
# several minutes, and solve-and-simulate costs about double a solve, so 2:00:00 has
# room for all three stages. chunked_seq is the slow one -- four sub-model solves per
# call, each with its own compilation -- so check a real log before shrinking this.
#
# Sizing --mem: host memory, not device. The chunked modes hold the sub-models plus
# the assembled solution, and chunked_parallel holds several blocks at once, so keep
# this well above what the pooled run needs.

module add cuda/12.4.1

python run_infrastructure_benchmark.py
