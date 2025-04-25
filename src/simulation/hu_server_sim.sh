#!/bin/bash

#SBATCH --job-name=pol_unc
#SBATCH --mail-user=maximilian.blesch@hu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH --mem=50GB
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --qos=standard

module add cuda/12.4.1

python run_model_fit_sim.py
