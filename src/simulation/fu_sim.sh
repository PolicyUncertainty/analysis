#!/bin/bash

#SBATCH --job-name=pol_unc_est
#SBATCH --mail-user=mblesch@zedat.fu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=80GB
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=standard

module add Python/3.12.3-GCCcore-13.3.0
module add CUDA/12.0.0

source ~/virts/bin/activate

#python run_model_fit_sim.py
python run_cf_sra_increase.py
python run_cf_debias.py
python run_cf_commitment.py
python run_cf_announcement_timing.py
