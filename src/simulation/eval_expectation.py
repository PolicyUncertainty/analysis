# %%
# Set paths of project
import pandas as pd
from matplotlib import pyplot as plt

from set_paths import create_path_dict

path_dict = create_path_dict()
import os

from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)

import pickle as pkl

# Import jax and set jax to work with 64bit
import jax
import numpy as np

from set_styles import get_figsize
from simulation.sim_tools.calc_aggregate_results import (
    calc_average_retirement_age,
    expected_lifetime_income,
)
from simulation.sim_tools.simulate_exp import simulate_exp

# %%
# Set specifications
model_name = specs["model_name"]
load_sol_model = True  # informed state as type
load_unc_solution = None  # baseline solution conntainer

# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

fixed_states = {
    "period": 0,
    "education": 0,
    "lagged_choice": 1,
    "sex": 0,
    "health": 0,
    "assets_begin_of_period": 6,
    "experience": 0.5,
    "partner_state": 0,
    "job_offer": 1,
}
fig, axs = plt.subplots(1, 2, figsize=get_figsize(1, 2))
fig2, axs2 = plt.subplots(1, 2, figsize=get_figsize(1, 2))

SRA_grid = np.arange(65, 71, 1)
for informed, informed_label in enumerate(["Uninformed", "Informed"]):
    ax = axs[informed]
    ax2 = axs2[informed]
    ax.set_title(f"Retirement Age - {informed_label}")
    ax.set_xlabel("SRA")
    ax.set_ylabel("Retirement Age")

    ax2.set_title(f"Expected Lifetime Income - {informed_label}")
    ax2.set_xlabel("SRA")
    ax2.set_ylabel("Expected Lifetime Income")
    for subj_unc in [True, False]:
        if subj_unc:
            exp_label = "expected reform"
        else:
            exp_label = "known reform"

        exp_ret_age = np.zeros_like(SRA_grid, dtype=float)
        exp_income = np.zeros_like(SRA_grid, dtype=float)
        for id_SRA, SRA in enumerate(SRA_grid):
            policy_state = int((SRA - 65) / 0.25)
            state = {**fixed_states, "policy_state": policy_state, "informed": informed}

            df = simulate_exp(
                initial_state=state,
                n_multiply=10_000,
                path_dict=path_dict,
                params=params,
                subj_unc=subj_unc,
                custom_resolution_age=None,
                model_name=model_name,
                solution_exists=load_unc_solution,
                sol_model_exists=load_sol_model,
            )
            exp_ret_age[id_SRA] = calc_average_retirement_age(df)
            exp_income[id_SRA] = expected_lifetime_income(df, specs)

        ax.plot(SRA_grid, exp_ret_age, label=exp_label)
        ax2.plot(SRA_grid, exp_income, label=exp_label)

    ax.legend()
    ax2.legend()

plot_folder = path_dict["simulation_plots"] + model_name + "/"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
# Save both plots as png and pdf
for figure, figure_label in zip([fig, fig2], ["retirement_age_", "expected_income_"]):
    figure.savefig(plot_folder + figure_label + ".png", bbox_inches="tight")
    figure.savefig(plot_folder + figure_label + ".pdf", bbox_inches="tight")
