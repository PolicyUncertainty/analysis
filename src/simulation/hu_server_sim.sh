#!/bin/bash

#SBATCH --job-name=sim_pol
#SBATCH --mail-user=maximilian.blesch@hu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:30:00
#SBATCH --mem=120GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --qos=standard

module add cuda/12.4.1


# python run_model_fit_sim.py
# python run_cf_debias.py
python run_baseline_expectation.py
# python run_cf_sra_increase.py
# python run_plots_and_tables.py
# python run_baseline_expectation.py
# python run_cf_sra_increase.py
# python run_plots_cf.py
# python run_eval_expectation_graphs.py
# python run_illustration.py
# python run_baseline.py
# python run_figures.py
# python run_cf_commitment.py
# python run_cf_announcement_timing.py
