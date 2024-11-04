#!/bin/bash

#SBATCH --job-name=pol_unc_est
#SBATCH --mail-user=mblesch@zedat.fu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=15GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard

module add Python/3.12.3-GCCcore-13.3.0
module add CUDA/12.0.0

python run_model_est.py
