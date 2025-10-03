#!/bin/bash

#SBATCH --job-name=pol_unc
#SBATCH --mail-user=maximilian.blesch@hu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=9:00:00
#SBATCH --mem=50GB
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100nvl:2
#SBATCH --qos=standard

module add cuda/12.8.1

#python run_msm.py
python plot_moments.py
