#!/bin/bash

#SBATCH --job-name=pol_unc_est
#SBATCH --mail-user=mblesch@zedat.fu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=15GB
#SBATCH --qos=standard

module add Anaconda3/2022.05
module add CUDA/12.0.0

module add Python/3.12.3-GCCcore-13.3.0
source ~/virts/bin/activate

conda activate policy_uncertainty

python run_estimation.py
