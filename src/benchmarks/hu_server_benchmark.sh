#!/bin/bash

#SBATCH --job-name=dcegm_bench
#SBATCH --mail-user=maximilian.blesch@hu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=120GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10080gb:1
#SBATCH --qos=standard

module add cuda/12.4.1

python run_infrastructure_benchmark.py
