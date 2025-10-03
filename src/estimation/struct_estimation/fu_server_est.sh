#!/bin/bash

#SBATCH --job-name=pol_est
#SBATCH --mail-user=mblesch@zedat.fu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=7:00:00
#SBATCH --mem=80GB
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=standard

module add Python/3.12.3-GCCcore-13.3.0
module add CUDA/12.0.0

source ~/virts/bin/activate


# python run_model_est.py
python run_fit_plot.py
#python ../../simulation/run_model_fit_sim.py
