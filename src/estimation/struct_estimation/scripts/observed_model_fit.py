import os

import pandas as pd

from set_styles import get_figsize, set_colors

JET_COLOR_MAP, LINE_STYLES = set_colors()
import matplotlib.pyplot as plt
import numpy as np

from estimation.struct_estimation.scripts.estimate_setup import (
    filter_data_by_type,
    generate_print_func,
)
from model_code.specify_model import specify_and_solve_model, specify_type_grids
from model_code.stochastic_processes.health_transition import (
    calc_disability_probability,
)
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from model_code.transform_data_from_model import (
    calc_choice_probs_for_df,
    calc_unobserved_choice_probs_for_df,
    load_scale_and_correct_data,
)
from set_styles import get_figsize, set_colors, set_plot_defaults


def create_fit_plots(
    path_dict,
    specs,
    params,
    model_name,
    load_sol_model,
    load_solution,
    load_data_from_sol,
    sex_type="all",
    edu_type="all",
    util_type="add",
    skip_model_plots=False,
):

    folder_name_plots = path_dict["estimation_plots"] + model_name + "/"
    # Check if folder exists, if not create it
    if not os.path.exists(folder_name_plots):
        os.makedirs(folder_name_plots)

    generate_print_func(params.keys(), specs)(params)

    plot_job_offers(
        save_folder=folder_name_plots,
        params=params,
        specs=specs,
    )

    plot_disability_probability(
        params=params,
        specs=specs,
        save_folder=folder_name_plots,
    )

    if skip_model_plots:
        return None

    if load_data_from_sol:
        data_decision = pd.read_csv(
            folder_name_plots + "data_with_probs.csv", index_col=0
        )

    else:

        data_decision = create_df_with_probs(
            path_dict=path_dict,
            params=params,
            model_name=model_name,
            load_sol_model=load_sol_model,
            load_solution=load_solution,
            edu_type=edu_type,
            sex_type=sex_type,
            util_type=util_type,
        )

        data_decision.to_csv(folder_name_plots + "data_with_probs.csv")

    specs["sex_grid"], specs["education_grid"] = specify_type_grids(
        sex_type=sex_type,
        edu_type=edu_type,
    )

    plot_ret_fit_age(
        specs=specs,
        data_decision=data_decision,
        save_folder=folder_name_plots,
    )
    plot_life_cycle_choice_probs_health(
        specs=specs,
        data_decision=data_decision,
        save_folder=folder_name_plots,
    )

    plot_life_cycle_choice_probs(
        specs=specs,
        data_decision=data_decision,
        save_folder=folder_name_plots,
    )

    plot_retirement_fit(
        specs=specs,
        data_decision=data_decision,
        save_folder=folder_name_plots,
    )

    plot_life_cycle_choice_probs_paper(
        specs=specs,
        data_decision=data_decision,
        save_folder=folder_name_plots,
    )

    # print_choice_probs_by_group(df=data_decision, specs=specs, path_dict=path_dict)

    # plt.show()
    # plt.close("all")


def plot_job_offers(
    save_folder,
    params,
    specs,
):
    # Initialize colors
    colors, _ = set_colors()

    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=get_figsize(2, 2))

    # Create age array from start_age to max_ret_age
    ages = np.arange(specs["start_age"], specs["max_ret_age"] + 1)
    periods = ages - specs["start_age"]

    # Iterate over sex and education combinations
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            # Select the appropriate subplot
            ax = axs[sex_var, edu_var]

            # Initialize arrays to store probabilities
            good_health_probs = []
            bad_health_probs = []

            # Calculate probabilities for each age
            for period in periods:
                # Good health (health=0 typically)
                prob_good = job_offer_process_transition(
                    params=params,
                    sex=sex_var,
                    health=np.array(specs["good_health_var"]),
                    policy_state=np.array(0),  # dummy, not used in calculation
                    model_specs=specs,
                    education=edu_var,
                    period=period,
                    choice=1,  # unemployment choice for job finding probability
                )
                good_health_probs.append(prob_good[1])  # probability of job offer

                # Bad health (health=1 typically)
                prob_bad = job_offer_process_transition(
                    params=params,
                    sex=sex_var,
                    health=np.array(specs["bad_health_var"]),
                    policy_state=np.array(0),  # dummy, not used in calculation
                    model_specs=specs,
                    education=edu_var,
                    period=period,
                    choice=1,
                )
                bad_health_probs.append(prob_bad[1])

            # Plot the two curves
            ax.plot(
                ages,
                good_health_probs,
                label="Good Health",
                color=colors[0],
                linewidth=2,
            )
            ax.plot(
                ages,
                bad_health_probs,
                label="Bad Health",
                color=colors[1],
                linewidth=2,
                linestyle="--",
            )

            # Formatting
            ax.set_title(f"{sex_label} - {edu_label}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Job Offer Probability")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 0.5])

    fig.tight_layout()
    fig.savefig(
        save_folder + "job_offer_probabilities.png",
        transparent=True,
        dpi=300,
    )
    fig.savefig(
        save_folder + "job_offer_probabilities.pdf",
        transparent=True,
        dpi=300,
    )


def plot_disability_probability(params, specs, save_folder):
    """Plot disability probability conditional on bad health by age, education and sex.

    Parameters
    ----------
    params : dict
        Model parameters including disability logit coefficients
    specs : dict
        Model specifications including age ranges and labels
    save_folder : str
        Path to folder where plots should be saved
    show : bool, default False
        Whether to display the plot
    """
    # Initialize colors
    colors, _ = set_colors()

    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=get_figsize(2, 2))

    # Create age array from start_age to max_ret_age
    ages = np.arange(specs["start_age"], specs["max_ret_age"] + 1)
    periods = ages - specs["start_age"]

    # Iterate over sex and education combinations
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            # Select the appropriate subplot
            ax = axs[sex_var, edu_var]

            # Initialize array to store disability probabilities
            disability_probs = []

            # Calculate disability probability for each age
            # (conditional on being in bad health)
            for period in periods:
                prob_disability = calc_disability_probability(
                    params=params,
                    sex=sex_var,
                    education=edu_var,
                    period=period,
                    model_specs=specs,
                )
                disability_probs.append(prob_disability)

            # Plot the disability probability curve
            ax.plot(
                ages,
                disability_probs,
                color=colors[sex_var * 2 + edu_var],
                linewidth=2,
                label="Disability Prob.",
            )

            # Formatting
            ax.set_title(f"{sex_label} - {edu_label}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Disability Probability")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.1])

    plt.tight_layout()
    fig.savefig(save_folder + "disability_prob_by_type.pdf", bbox_inches="tight")
    fig.savefig(
        save_folder + "disability_prob_by_type.png", bbox_inches="tight", dpi=300
    )


def create_df_with_probs(
    path_dict,
    params,
    model_name,
    load_sol_model,
    load_solution,
    edu_type="all",
    sex_type="all",
    util_type="add",
    unobs_choice_probs=False,
):
    model_solved = specify_and_solve_model(
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        file_append=model_name,
        load_model=load_sol_model,
        load_solution=load_solution,
        sim_specs=None,
        edu_type=edu_type,
        sex_type=sex_type,
        util_type=util_type,
        debug_info="all",
    )

    data_decision = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_solved
    )

    data_decision = filter_data_by_type(
        df=data_decision, sex_type=sex_type, edu_type=edu_type
    )

    data_decision = calc_choice_probs_for_df(
        df=data_decision, params=params, model_solved=model_solved
    )
    data_decision["SRA_diff"] = (
        data_decision["age"] - data_decision["policy_state_value"]
    )
    if unobs_choice_probs:
        data_decision = calc_unobserved_choice_probs_for_df(
            df=data_decision, params=params, model_solved=model_solved
        )
    return data_decision


def plot_ret_fit_age(
    specs,
    data_decision,
    save_folder,
):
    df_int = data_decision[data_decision["age"] < 75].copy()

    old_ages = np.arange(60, 72)

    choice_share_labels = ["Choice Share Men", "Choice Share Women"]
    health_labels = ["Not good health", "Good health"]
    good_health_mask = df_int["health"] == 0
    for sex_var in specs["sex_grid"]:
        sex_label = specs["sex_labels"][sex_var]
        if sex_var == 0:
            n_choices = 3
        else:
            n_choices = 4
        fig, axes = plt.subplots(
            ncols=n_choices, figsize=get_figsize(ncols=n_choices, nrows=1)
        )
        count = -1
        for edu_var in specs["education_grid"]:
            edu_label = specs["education_labels"][edu_var]

            for health_label in health_labels:
                count += 1
                if health_label == "Good health":
                    mask = good_health_mask
                else:
                    mask = ~good_health_mask
                data_subset = df_int[
                    (df_int["education"] == edu_var) & (df_int["sex"] == sex_var) & mask
                ]
                all_choice_shares_obs = (
                    data_subset.groupby(["age"])["choice"]
                    .value_counts(normalize=True)
                    .unstack()
                )
                labels = specs["choice_labels"]
                for choice in range(specs["n_choices"]):

                    choice_shares_predicted = data_subset.groupby(["age"])[
                        f"choice_{choice}"
                    ].mean()

                    # Only plot if we are not in men and part-time
                    men_and_part_time = (sex_var == 0) and (choice == 2)
                    if not men_and_part_time:
                        men_and_full_time = (sex_var == 0) and (choice == 3)
                        if men_and_full_time:
                            ax = axes[choice - 1]
                        else:
                            ax = axes[choice]

                        ax.plot(
                            choice_shares_predicted.reindex(old_ages),
                            label=f"Pred. {edu_label} - {health_label}",
                            color=JET_COLOR_MAP[count],
                        )
                        ax.plot(
                            all_choice_shares_obs[choice].reindex(old_ages),
                            label=f"Obs. {edu_label} - {health_label}",
                            color=JET_COLOR_MAP[count],
                            linestyle="--",
                        )

                        ax.set_ylim([-0.05, 1.05])
                        ax.set_title(f"{labels[choice]}")
                        ax.set_xlabel("Age")

        axes[1].legend(loc="upper left")
        axes[0].set_ylabel(choice_share_labels[sex_var])
        # Fig title
        fig.tight_layout()

        fig.savefig(  # fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")
            save_folder + f"ret_age_fit_{sex_label}.png",
            transparent=True,
            dpi=300,
        )
        fig.savefig(  # fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")
            save_folder + f"ret_age_fit_{sex_label}.pdf",
            transparent=True,
            dpi=300,
        )


def plot_life_cycle_choice_probs_health(
    specs,
    data_decision,
    save_folder,
):
    df_int = data_decision[data_decision["age"] < 75].copy()

    choice_share_labels = ["Choice Share Men", "Choice Share Women"]
    health_labels = ["Not good health", "Good health"]
    good_health_mask = df_int["health"] == 0
    for sex_var in specs["sex_grid"]:
        sex_label = specs["sex_labels"][sex_var]
        if sex_var == 0:
            n_choices = 3
        else:
            n_choices = 4
        fig, axes = plt.subplots(
            ncols=n_choices, figsize=get_figsize(ncols=n_choices, nrows=1)
        )
        count = -1
        for edu_var in specs["education_grid"]:
            edu_label = specs["education_labels"][edu_var]

            for health_label in health_labels:
                count += 1
                if health_label == "Good health":
                    mask = good_health_mask
                else:
                    mask = ~good_health_mask
                data_subset = df_int[
                    (df_int["education"] == edu_var) & (df_int["sex"] == sex_var) & mask
                ]
                all_choice_shares_obs = (
                    data_subset.groupby(["age"])["choice"]
                    .value_counts(normalize=True)
                    .unstack()
                )

                labels = specs["choice_labels"]
                for choice in range(specs["n_choices"]):

                    choice_shares_predicted = data_subset.groupby(["age"])[
                        f"choice_{choice}"
                    ].mean()

                    # Only plot if we are not in men and part-time
                    men_and_part_time = (sex_var == 0) and (choice == 2)
                    if not men_and_part_time:
                        men_and_full_time = (sex_var == 0) and (choice == 3)
                        if men_and_full_time:
                            ax = axes[choice - 1]
                        else:
                            ax = axes[choice]

                        ax.plot(
                            choice_shares_predicted,
                            label=f"Pred. {edu_label} - {health_label}",
                            color=JET_COLOR_MAP[count],
                        )
                        ax.plot(
                            all_choice_shares_obs[choice],
                            label=f"Obs. {edu_label} - {health_label}",
                            color=JET_COLOR_MAP[count],
                            linestyle="--",
                        )

                        ax.set_ylim([-0.05, 1.05])
                        ax.set_title(f"{labels[choice]}")
                        ax.set_xlabel("Age")

        axes[1].legend(loc="upper left")
        axes[0].set_ylabel(choice_share_labels[sex_var])
        # Fig title
        fig.tight_layout()

        fig.savefig(  # fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")
            save_folder + f"observed_model_fit_health_{sex_label}.png",
            transparent=True,
            dpi=300,
        )
        fig.savefig(  # fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")
            save_folder + f"observed_model_fit_health_{sex_label}.pdf",
            transparent=True,
            dpi=300,
        )


def plot_life_cycle_choice_probs_paper(
    specs,
    data_decision,
    save_folder,
):
    set_plot_defaults()
    df_int = data_decision[data_decision["age"] < 75].copy()

    # Get a single figsize for all plots
    figsize = get_figsize(ncols=1, nrows=1)

    for sex_var in specs["sex_grid"]:
        sex_label = specs["sex_labels"][sex_var]

        labels = specs["choice_labels"]

        for choice in range(specs["n_choices"]):
            # Skip men and part-time combination
            men_and_part_time = (sex_var == 0) and (choice == 2)
            if men_and_part_time:
                continue

            # Create a new figure for this choice
            fig, ax = plt.subplots(figsize=figsize)

            for edu_var in specs["education_grid"]:
                edu_label = specs["education_labels"][edu_var]
                data_subset = df_int[
                    (df_int["education"] == edu_var) & (df_int["sex"] == sex_var)
                ]
                all_choice_shares_obs = (
                    data_subset.groupby(["age"])["choice"]
                    .value_counts(normalize=True)
                    .unstack()
                )

                choice_shares_predicted = data_subset.groupby(["age"])[
                    f"choice_{choice}"
                ].mean()

                lower_edu_label = edu_label.lower()

                ax.plot(
                    choice_shares_predicted,
                    label=f"pred. {lower_edu_label}",
                    color=JET_COLOR_MAP[edu_var],
                )
                ax.plot(
                    all_choice_shares_obs[choice],
                    label=f"obs. {lower_edu_label}",
                    color=JET_COLOR_MAP[edu_var],
                    linestyle="--",
                )

            ax.set_ylim([-0.05, 1.05])
            ax.set_xlabel("Age")
            ax.set_ylabel("Choice Share")

            # Only add legend to retirement plots (choice == 0)
            if choice == 0:
                ax.legend(loc="upper left")

            fig.tight_layout()

            # Save with choice name in filename
            choice_name = labels[choice].replace(" ", "_").lower()

            paper_plots_folder = save_folder + "paper_fits/"
            os.makedirs(paper_plots_folder, exist_ok=True)

            fig.savefig(
                paper_plots_folder
                + f"observed_model_fit_{sex_label}_{choice_name}.png",
                transparent=True,
                dpi=300,
            )
            plt.close(fig)


def plot_life_cycle_choice_probs(
    specs,
    data_decision,
    save_folder,
):
    df_int = data_decision[
        (data_decision["age"] < 75) & (data_decision["lagged_choice"] != 0)
    ].copy()

    choice_share_labels = ["Choice Share Men", "Choice Share Women"]
    for sex_var in specs["sex_grid"]:
        sex_label = specs["sex_labels"][sex_var]
        if sex_var == 0:
            n_choices = 3
        else:
            n_choices = 4
        fig, axes = plt.subplots(
            ncols=n_choices, figsize=get_figsize(ncols=n_choices, nrows=1)
        )
        for edu_var in specs["education_grid"]:
            edu_label = specs["education_labels"][edu_var]
            data_subset = df_int[
                (df_int["education"] == edu_var) & (df_int["sex"] == sex_var)
            ]
            all_choice_shares_obs = (
                data_subset.groupby(["age"])["choice"]
                .value_counts(normalize=True)
                .unstack()
            )

            labels = specs["choice_labels"]
            for choice in range(specs["n_choices"]):

                choice_shares_predicted = data_subset.groupby(["age"])[
                    f"choice_{choice}"
                ].mean()

                # Only plot if we are not in men and part-time
                men_and_part_time = (sex_var == 0) and (choice == 2)
                if not men_and_part_time:
                    men_and_full_time = (sex_var == 0) and (choice == 3)
                    if men_and_full_time:
                        ax = axes[choice - 1]
                    else:
                        ax = axes[choice]

                    ax.plot(
                        choice_shares_predicted,
                        label=f"Pred. {edu_label}",
                        color=JET_COLOR_MAP[edu_var],
                    )
                    ax.plot(
                        all_choice_shares_obs[choice],
                        label=f"Obs. {edu_label}",
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="--",
                    )

                    ax.set_ylim([-0.05, 1.05])
                    ax.set_title(f"{labels[choice]}")
                    ax.set_xlabel("Age")

        axes[1].legend(loc="upper left")
        axes[0].set_ylabel(choice_share_labels[sex_var])
        # Fig title
        fig.tight_layout()

        fig.savefig(  # fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")
            save_folder + f"observed_model_fit_{sex_label}.png",
            transparent=True,
            dpi=300,
        )
        fig.savefig(  # fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")
            save_folder + f"observed_model_fit_{sex_label}.pdf",
            transparent=True,
            dpi=300,
        )


def plot_retirement_fit(
    specs,
    data_decision,
    save_folder,
):
    diffs = np.arange(-4, 1.25, 0.25)
    pos = np.arange(0, len(diffs))

    not_retired_mask = data_decision["lagged_choice"] != 0
    data_subset = data_decision[not_retired_mask].copy()

    choices_shares_obs = (
        data_subset.groupby(["sex", "SRA_diff"])["choice"]
        .value_counts()
        .unstack()
        .fillna(0.0)
    )
    rename_map = {
        key: choice_label for key, choice_label in enumerate(specs["choice_labels"])
    }

    choices_shares_obs.rename(rename_map, axis=1, inplace=True)

    # Create a stacked bar plot with each choice sum on top of each other
    choice_shares_est = data_subset.groupby(["sex", "SRA_diff"])[
        [f"choice_{choice}" for choice in range(specs["n_choices"])]
    ].sum()
    rename_map_2 = {
        f"choice_{key}": choice_label
        for key, choice_label in enumerate(specs["choice_labels"])
    }
    choice_shares_est.rename(rename_map_2, axis=1, inplace=True)

    fig, axs = plt.subplots(nrows=2, figsize=(14, 8))

    for sex_var in specs["sex_grid"]:
        sex_label = specs["sex_labels"][sex_var]

        # if sex_var == 0:
        #     choice_shares_est.drop("Part-time", axis=1, inplace=True)

        choice_shares_est_to_plot = choice_shares_est.loc[sex_var].reindex(diffs)
        choices_shares_obs_to_plot = choices_shares_obs.loc[sex_var].reindex(diffs)

        # Plotting the retirement choice shares as bars (observed as hatched)
        ax = axs[sex_var]
        choice_shares_est_to_plot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=JET_COLOR_MAP[: specs["n_choices"]],
            width=0.4,
            position=0,
            hatch="/",
        )
        choices_shares_obs_to_plot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=JET_COLOR_MAP[: specs["n_choices"]],
            width=0.4,
            position=1,
            label="Observed",
        )

    # Set x-axis labels and ticks
    axs[0].set_xlabel("Difference to SRA")
    axs[0].set_xticks(pos)
    axs[0].set_xticklabels(diffs)
    axs[1].set_xlabel("Difference to SRA")
    axs[1].set_xticks(pos)
    axs[1].set_xticklabels(diffs)

    axs[0].set_title("Men")
    axs[1].set_title("Women")
    axs[0].set_ylabel("Choice Share")
    fig.tight_layout()
    fig.savefig(
        save_folder + f"retirement_fit.png",
        transparent=True,
        dpi=300,
    )
    fig.savefig(
        save_folder + f"retirement_fit.pdf",
        transparent=True,
        dpi=300,
    )
