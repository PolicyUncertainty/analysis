#!/bin/bash

#SBATCH --job-name=est_pol
#SBATCH --mail-user=maximilian.blesch@hu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --mem=80GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --qos=standard

module add cuda/12.4.1

# python run_model_est.py
python run_fit_plot.py
#python ../../simulation/run_model_fit_sim.py
# python grid_search.py
