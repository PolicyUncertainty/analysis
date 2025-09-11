#!/bin/bash

#SBATCH --job-name=soepy_estimation
#SBATCH --mail-user=mblesch@zedat.fu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=400
#SBATCH --time=8:00:00
#SBATCH --qos=standard

module add Python/3.12.3-GCCcore-13.3.0

source ~/virts/bin/activate

#python run_fit_plot.py
#python ../../simulation/run_model_fit_sim.py
python run_model_est.py
