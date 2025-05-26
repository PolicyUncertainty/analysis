import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from export_results.figures.color_map import JET_COLOR_MAP
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario


def plot_sim_vars(
    path_dict,
    specs,
    params,
    model_name,
    sim_var,
    plot_dead=False,
    load_df=True,
    load_solution=True,
    load_sol_model=True,
):
    # Simulate baseline with subjective belief
    data_sim = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        announcement_age=None,
        subj_unc=True,
        custom_resolution_age=None,
        SRA_at_start=67,
        SRA_at_retirement=67,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_solution,
        sol_model_exists=load_sol_model,
    ).reset_index()

    data_sim["age"] = data_sim["period"] + specs["start_age"]

    if not plot_dead:
        data_sim = data_sim[data_sim["health"] != 2]
        data_sim = data_sim[data_sim["age"] < specs["end_age"]]

    fig, axs = plt.subplots(ncols=specs["n_education_types"])
    # Also generate an aggregate graph
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axs[edu_var]
            mask_sim = (data_sim["sex"] == sex_var) & (data_sim["education"] == edu_var)
            data_sim_edu = data_sim[mask_sim]

            average_var = data_sim_edu.groupby("age")[sim_var].mean()

            ax.plot(average_var, color=JET_COLOR_MAP[sex_var], label=f"{sex_label}")

            ax.set_title(f"{edu_label}")
            ax.legend()

    fig.suptitle(f"{sim_var}")
