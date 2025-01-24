#!/bin/bash

#SBATCH --job-name=pol_unc_est
#SBATCH --mail-user=mblesch@zedat.fu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=30GB
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:a5000:1
#SBATCH --qos=standard

module add Python/3.12.3-GCCcore-13.3.0
module add CUDA/12.0.0

source ~/virts/bin/activate

python run_counterfactual_1.py
python run_counterfactual_2.py
