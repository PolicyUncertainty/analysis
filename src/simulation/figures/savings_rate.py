import os

import numpy as np
from matplotlib import pyplot as plt

from set_styles import get_figsize
from simulation.figures.simulated_model_fit import JET_COLOR_MAP
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario


def plot_savings(
    path_dict,
    specs,
    params,
    model_name,
    file_name,
    load_df=True,
    load_solution=True,
    load_sol_model=True,
    util_type="add"
):
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
        util_type=util_type
    )

    data_sim = data_sim.reset_index()

    data_sim["age"] = data_sim["period"] + specs["start_age"]

    fig, axs = plt.subplots(
        ncols=specs["n_education_types"],
        figsize=get_figsize(ncols=specs["n_education_types"]),
    )
    # Also generate an aggregate graph
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axs[edu_var]
            mask_sim = (data_sim["sex"] == sex_var) & (data_sim["education"] == edu_var)
            data_sim_edu = data_sim[mask_sim]

            ages = np.arange(specs["start_age"] + 1, 65)

            mean_savings = data_sim_edu.groupby("age")["savings_dec"].mean().loc[ages]
            mean_income = data_sim_edu.groupby("age")["total_income"].mean().loc[ages]
            savings_rate = mean_savings / mean_income

            ax.plot(
                ages,
                savings_rate,
                label=f"Savings rate {sex_label}",
                color=JET_COLOR_MAP[sex_var],
            )
            ax.set_title(f"{edu_label}")
    axs[0].legend()

    plot_folder = path_dict["simulation_plots"] + model_name + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    if file_name is not None:
        fig.savefig(plot_folder + f"{file_name}.png", transparent=True, dpi=300)
        fig.savefig(plot_folder + f"{file_name}.pdf", dpi=300)
