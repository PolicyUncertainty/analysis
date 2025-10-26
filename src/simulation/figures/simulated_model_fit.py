import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import get_figsize, set_colors, set_plot_defaults

JET_COLOR_MAP, LINE_STYLES = set_colors()
from model_code.transform_data_from_model import load_scale_and_correct_data
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario


def create_paper_wealth_fit(
    path_dict,
    specs,
    params,
    model_name,
    load_df=True,
    load_solution=True,
    load_sol_model=True,
    util_type="add",
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
        util_type=util_type,
    )

    data_sim = data_sim.reset_index()

    data_decision = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_solved
    )

    set_plot_defaults()
    figsize = get_figsize(ncols=1, nrows=1)

    fig, ax = plt.subplots(figsize=figsize)
    max_wealth = 50
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        mask_sim = data_sim["education"] == edu_var
        data_sim_edu = data_sim[mask_sim]
        mask_obs = data_decision["education"] == edu_var
        data_decision_edu = data_decision[mask_obs]

        ages = np.arange(specs["start_age"] + 1, 90)

        average_wealth_sim = (
            data_sim_edu.groupby("age")["assets_begin_of_period"].median().loc[ages]
        ) * 10
        average_wealth_obs = (
            data_decision_edu.groupby("age")["assets_begin_of_period"]
            .median()
            .loc[ages]
        ) * 10
        max_wealth = max(
            max_wealth,
            average_wealth_sim.max(),
            average_wealth_obs.max(),
        )

        lower_edu_label = edu_label.lower()

        ax.plot(
            ages,
            average_wealth_sim,
            label=f"sim. {lower_edu_label}",
            color=JET_COLOR_MAP[edu_var],
        )
        ax.plot(
            ages,
            average_wealth_obs,
            label=f"obs. {lower_edu_label}",
            ls="--",
            color=JET_COLOR_MAP[edu_var],
        )

    ax.legend(frameon=False)
    ax.set_ylim([0, max_wealth * 1.4])

    plot_folder = path_dict["simulation_plots"] + model_name + "/paper_plots/"
    os.makedirs(plot_folder, exist_ok=True)

    fig.savefig(plot_folder + f"wealth_fit.png", transparent=True, dpi=300)
    # fig.savefig(plot_folder + f"{file_name}.pdf", dpi=300)


def plot_quantiles(
    path_dict,
    specs,
    params,
    model_name,
    quantiles,
    sim_col_name,
    obs_col_name,
    file_name,
    load_df=True,
    load_solution=True,
    load_sol_model=True,
    util_type="add",
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
        util_type=util_type,
    )

    data_sim = data_sim.reset_index()

    data_decision = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_solved
    )

    data_sim["age"] = data_sim["period"] + specs["start_age"]

    fig, axs = plt.subplots(
        ncols=specs["n_education_types"],
        figsize=get_figsize(ncols=specs["n_education_types"]),
    )
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

            ages = np.arange(specs["start_age"] + 1, 90)

            for quant in quantiles:
                average_wealth_sim = (
                    data_sim_edu.groupby("age")[sim_col_name].quantile(quant).loc[ages]
                )
                average_wealth_obs = (
                    data_decision_edu.groupby("age")[obs_col_name]
                    .quantile(quant)
                    .loc[ages]
                )
                max_wealth = max(
                    max_wealth,
                    average_wealth_sim.max(),
                    average_wealth_obs.max(),
                )

                if np.allclose(quant, 0.5):
                    name = "Median"
                else:
                    name = f"{int(quant*100)}th Perc."

                ax.plot(
                    ages,
                    average_wealth_sim,
                    label=f"Sim. {name} {sex_label}",
                    color=JET_COLOR_MAP[sex_var],
                )
                ax.plot(
                    ages,
                    average_wealth_obs,
                    label=f"Obs. {name} {sex_label}",
                    ls="--",
                    color=JET_COLOR_MAP[sex_var],
                )
            ax.set_title(f"{edu_label}")
    axs[0].legend()
    for edu in range(specs["n_education_types"]):
        axs[edu].set_ylim([0, max_wealth * 1.1])

    plot_folder = path_dict["simulation_plots"] + model_name + "/"

    if file_name is not None:
        fig.savefig(plot_folder + f"{file_name}.png", transparent=True, dpi=300)
        fig.savefig(plot_folder + f"{file_name}.pdf", dpi=300)


def plot_choice_shares_single(
    path_dict,
    specs,
    params,
    model_name,
    file_name=None,
    load_df=True,
    load_solution=True,
    load_sol_model=True,
    util_type="add",
):
    # alpha_belief = float(np.loadtxt(
    # path_dict["est_results"] + "exp_val_params.txt"))

    # Simulate baseline with subjective belief
    data_sim, _ = solve_and_simulate_scenario(
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
        util_type=util_type,
    )

    data_sim = data_sim.reset_index()
    data_decision = pd.read_csv(path_dict["struct_est_sample"])

    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_sim["age"] = data_sim["period"] + specs["start_age"]

    fig, axes = plt.subplots(
        2, specs["n_choices"], figsize=get_figsize(ncols=specs["n_choices"], nrows=2)
    )
    for sex, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            data_sim_restr = data_sim[
                (data_sim["sex"] == sex) & (data_sim["education"] == edu_var)
            ]
            data_decision_restr = data_decision[
                (data_decision["sex"] == sex) & (data_decision["education"] == edu_var)
            ]

            choice_shares_sim = (
                data_sim_restr.groupby(["age"])["choice"]
                .value_counts(normalize=True)
                .unstack()
            )
            choice_shares_obs = (
                data_decision_restr.groupby(["age"])["choice"]
                .value_counts(normalize=True)
                .unstack()
            )
            if sex == 0:
                choice_range = [0, 1, 3]
            else:
                choice_range = range(4)

            for choice in choice_range:
                ax = axes[sex, choice]
                choice_share_sim = choice_shares_sim[choice]
                choice_share_obs = choice_shares_obs[choice]
                ax.plot(
                    choice_share_sim,
                    label=f"Simulated; {edu_label}",
                    color=JET_COLOR_MAP[edu_var],
                )
                ax.plot(
                    choice_share_obs,
                    label=f"Observed; {edu_label}",
                    ls="--",
                    color=JET_COLOR_MAP[edu_var],
                )
                choice_label = specs["choice_labels"][choice]
                ax.set_ylim([0, 1])
                if sex == 0:
                    ax.set_title(f"{choice_label}")
                    if choice == 1:
                        ax.legend()

        axes[sex, 0].set_ylabel(f"{sex_label}; Choice shares")

    plot_folder = path_dict["simulation_plots"] + model_name + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    if file_name is not None:
        fig.savefig(plot_folder + f"{file_name}.png", transparent=True, dpi=300)
        fig.savefig(plot_folder + f"{file_name}.pdf", dpi=300)


def plot_states(
    path_dict,
    specs,
    params,
    model_name,
    load_df=True,
    load_solution=True,
    load_sol_model=True,
    util_type="add",
):
    # Simulate baseline with subjective belief
    data_sim, model_solved = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        announcement_age=None,
        SRA_at_retirement=67,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_solution,
        sol_model_exists=load_sol_model,
        util_type=util_type,
    )

    data_sim = data_sim.reset_index()

    data_decision = pd.read_csv(path_dict["struct_est_sample"])

    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_sim["age"] = data_sim["period"] + specs["start_age"]

    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_sim["age"] = data_sim["period"] + specs["start_age"]

    grouped_obs = data_decision.groupby(["sex", "education", "age"])
    grouped_sim = data_sim.groupby(["sex", "education", "age"])

    grouped_obs["experience"].mean().loc[(0, 1, slice(None))].plot()
    grouped_sim["exp_years"].mean().loc[(0, 1, slice(None))].plot()
    plt.legend()
    plt.show()

    model_structure = model_solved.model_structure
    discrete_state_names = model_structure["discrete_states_names"]

    data_sim = data_sim[data_sim["health"] != 3]

    for state_name in discrete_state_names:
        if state_name in ["period", "informed", "policy_state"]:
            continue

        if state_name in ["education", "sex"]:
            obs_shares = data_decision.groupby(["age"])[state_name].value_counts(
                normalize=True
            )
            sim_shares = data_sim.groupby(["age"])[state_name].value_counts(
                normalize=True
            )

            fig, ax = plt.subplots()

            for state_val in sim_shares.index.get_level_values(1).unique():
                ax.plot(
                    obs_shares.loc[(slice(None), state_val)],
                    # label=f"Observed; {state_name} = {state_val}",
                    ls="--",
                    color=JET_COLOR_MAP[state_val],
                )
                ax.plot(
                    sim_shares.loc[(slice(None), state_val)],
                    label=f"{state_val}",
                    color=JET_COLOR_MAP[state_val],
                )
                ax.legend()
                ax.set_title(state_name)

        else:
            fig, axs = plt.subplots(
                nrows=specs["n_sexes"], ncols=specs["n_education_types"]
            )
            for sex_var, sex_label in enumerate(specs["sex_labels"]):
                for edu_var, edu_label in enumerate(specs["education_labels"]):
                    ax = axs[sex_var, edu_var]
                    df_type_sim = data_sim[
                        (data_sim["sex"] == sex_var)
                        & (data_sim["education"] == edu_var)
                    ]
                    df_type_obs = data_decision[
                        (data_decision["sex"] == sex_var)
                        & (data_decision["education"] == edu_var)
                    ]

                    sim_shares = df_type_sim.groupby(["age"])[state_name].value_counts(
                        normalize=True
                    )

                    if state_name == "health":
                        obs_shares = df_type_obs.groupby(["age"])[
                            "surveyed_health"
                        ].value_counts(normalize=True)
                        ages = sim_shares.index.get_level_values("age").unique()
                        full_index = pd.MultiIndex.from_product(
                            [ages, range(3)], names=["age", "health"]
                        )
                        sim_shares = sim_shares.reindex(full_index, fill_value=0)
                        sim_shares.loc[(slice(None), 1)] = (
                            sim_shares.loc[(slice(None), 1)].values
                            + sim_shares.loc[(slice(None), 2)].values
                        )

                    else:
                        obs_shares = df_type_obs.groupby(["age"])[
                            state_name
                        ].value_counts(normalize=True)

                    for state_val in sim_shares.index.get_level_values(1).unique():
                        if (state_name == "health") and (state_val == 2):
                            ax.plot(
                                sim_shares.loc[(slice(None), state_val)],
                                label=f"{state_val}",
                                color=JET_COLOR_MAP[state_val],
                            )
                            continue
                        else:
                            ax.plot(
                                obs_shares.loc[(slice(None), state_val)],
                                # label=f"Observed; {state_name} = {state_val}",
                                ls="--",
                                color=JET_COLOR_MAP[state_val],
                            )
                        ax.plot(
                            sim_shares.loc[(slice(None), state_val)],
                            label=f"{state_val}",
                            color=JET_COLOR_MAP[state_val],
                        )
                    ax.legend()
                    ax.set_title(f"{sex_label}; {edu_label}")
            fig.suptitle(state_name)
            fig.savefig(
                path_dict["plots"] + f"{state_name}.png", transparent=True, dpi=300
            )
            fig.savefig(path_dict["plots"] + f"{state_name}.pdf", dpi=300)
