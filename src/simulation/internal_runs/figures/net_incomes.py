import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import set_colors

JET_COLOR_MAP, LINE_STYLES = set_colors()
from model_code.transform_data_from_model import load_scale_and_correct_data
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario


def net_incomes(
    path_dict,
    specs,
    params,
    model_name,
    file_name,
    load_df=True,
    load_solution=True,
    load_sol_model=True,
):
    sim_col_name = "gross_hh_income"
    obs_col_name = "yearly_wage"

    # Simulate baseline with subjective belief
    data_sim, model_solved = solve_and_simulate_scenario(
        announcement_age=None,
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        SRA_at_retirement=67,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_solution,
        sol_model_exists=load_sol_model,
    )

    data_sim = data_sim.reset_index()

    data_decision, _ = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_solved
    )
    data_decision = data_decision[(data_decision["choice"] == 3)]
    data_decision["yearly_wage"] = data_decision["monthly_wage"] * 12

    data_sim["age"] = data_sim["period"] + specs["start_age"]
    data_sim = data_sim[(data_sim["lagged_choice"] == 3)]

    fig, axs = plt.subplots(ncols=specs["n_education_types"])
    max_wealth = 5
    # Also generate an aggregate graph
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axs[edu_var]
            mask_sim = (data_sim["sex"] == sex_var) & (data_sim["education"] == edu_var)
            data_sim_edu = data_sim[mask_sim]
            mask_obs = (data_decision["sex"] == sex_var) & (
                data_decision["education"] == edu_var
            )
            data_decision_edu = data_decision[mask_obs]

            ages = np.arange(specs["start_age"] + 1, 60)

            average_income_sim = (
                data_sim_edu.groupby("age")[sim_col_name].mean().loc[ages]
            )
            average_wealth_obs = (
                data_decision_edu.groupby("age")[obs_col_name].mean().loc[ages]
            )
            max_wealth = max(
                max_wealth,
                average_income_sim.max(),
                average_wealth_obs.max(),
            )

            ax.plot(
                ages,
                average_income_sim,
                label=f"Sim. {sex_label}",
                color=JET_COLOR_MAP[sex_var],
            )
            ax.plot(
                ages,
                average_wealth_obs,
                label=f"Obs. {sex_label}",
                ls="--",
                color=JET_COLOR_MAP[sex_var],
            )
            ax.set_title(f"{edu_label}")
    axs[0].legend()
    for edu in range(specs["n_education_types"]):
        axs[edu].set_ylim([0, max_wealth * 1.1])
    if file_name is not None:
        fig.savefig(path_dict["plots"] + f"{file_name}.png", transparent=True, dpi=300)
