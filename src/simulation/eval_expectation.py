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
    expected_pension,
    expected_working_lifetime_income,
)
from simulation.sim_tools.simulate_exp import simulate_exp

# %%
# Set specifications
model_name = specs["model_name"]
load_sol_model = True  # informed state as type
load_solution = True  # baseline solution conntainer

# Load params
params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

fixed_states = {
    "period": 0,
    "lagged_choice": 1,
    "health": 0,
    "assets_begin_of_period": 6,
    "experience": 0.5,
    "partner_state": 0,
    "job_offer": 1,
}

type_list = []
type_vars = []
for sex_var, sex_label in enumerate(specs["sex_labels"]):
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        type_list += [f"{sex_label}_{edu_label}"]
        type_vars += [(sex_var, edu_var)]


n_types = len(type_list)
fig, axs = plt.subplots(n_types, 2, figsize=get_figsize(n_types, 2))
fig2, axs2 = plt.subplots(n_types, 2, figsize=get_figsize(n_types, 2))
fig3, axs3 = plt.subplots(n_types, 2, figsize=get_figsize(n_types, 2))
fig4, axs4 = plt.subplots(n_types, 2, figsize=get_figsize(n_types, 2))

df_dict = {}
SRA_grid = np.arange(67, 68)
n_grid = len(SRA_grid)
for subj_unc in [True, False]:
    model_solution = None
    for informed, informed_label in enumerate(["Uninformed", "Informed"]):
        axs[0, informed].set_title(f"Retirement Age - {informed_label}")
        axs2[0, informed].set_title(f"Expected Lifetime Income - {informed_label}")

        for id_type in range(n_types):
            print(
                "Eval expectation: ",
                subj_unc,
                informed_label,
                type_list[id_type],
                flush=True,
            )
            sex_var, edu_var = type_vars[id_type]
            type_label = type_list[id_type]

            exp_ret_age = np.zeros(n_grid, dtype=float)
            exp_income = np.zeros(n_grid, dtype=float)
            exp_pension = np.zeros(n_grid, dtype=float)
            exp_work_income = np.zeros(n_grid, dtype=float)
            for id_SRA, SRA in enumerate(SRA_grid):
                print(SRA, flush=True)
                policy_state = int((SRA - 65) / 0.25)
                state = {
                    **fixed_states,
                    "policy_state": policy_state,
                    "informed": informed,
                    "sex": sex_var,
                    "education": edu_var,
                }

                df, model_solution = simulate_exp(
                    initial_state=state,
                    n_multiply=1_000,
                    path_dict=path_dict,
                    params=params,
                    subj_unc=subj_unc,
                    custom_resolution_age=None,
                    model_name=model_name,
                    solution_exists=load_solution,
                    sol_model_exists=load_sol_model,
                    model_solution=model_solution,
                )
                exp_ret_age[id_SRA] = calc_average_retirement_age(df)
                exp_income[id_SRA] = expected_lifetime_income(df, specs)
                exp_pension[id_SRA] = expected_pension(df)
                exp_work_income[id_SRA] = expected_working_lifetime_income(df, specs)
                # df_dict[SRA] = df

            if subj_unc:
                exp_label = "expected reform"
            else:
                exp_label = "known reform"

            ax = axs[id_type, informed]
            ax.set_ylabel(f"Age - {type_label}")
            ax.plot(SRA_grid, exp_ret_age, label=exp_label)

            ax2 = axs2[id_type, informed]
            ax2.set_ylabel(f"Income - {type_label}")
            ax2.plot(SRA_grid, exp_income, label=exp_label)

            ax3 = axs3[id_type, informed]
            ax3.set_ylabel(f"Pension - {type_label}")
            ax3.plot(SRA_grid, exp_pension, label=exp_label)

            ax4 = axs4[id_type, informed]
            ax4.set_ylabel(f"Work Income - {type_label}")
            ax4.plot(SRA_grid, exp_work_income, label=exp_label)

for fig_axs in [axs, axs2, axs3, axs4]:
    fig_axs[0, 0].legend()
    fig_axs[-1, 0].set_xlabel("SRA")


plot_folder = path_dict["simulation_plots"] + model_name + "/"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
# Save both plots as png and pdf
for figure, figure_label in zip(
    [fig, fig2, fig3, fig4],
    [
        "retirement_age_",
        "expected_income_",
        "expected_pension_",
        "expected_work_income_",
    ],
):
    figure.savefig(plot_folder + figure_label + ".png", bbox_inches="tight")
    figure.savefig(plot_folder + figure_label + ".pdf", bbox_inches="tight")
