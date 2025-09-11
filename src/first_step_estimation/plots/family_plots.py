# Description: This file contains plotting functions for family transition estimation results.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import set_colors, get_figsize


def plot_family_transition_results(path_dict, specs, show=False, save=False):
    """Plot family transition estimation results including transition matrices and predicted vs empirical shares."""

    # Load the transition matrix results
    full_df = pd.read_csv(
        path_dict["first_step_results"] + "partner_transition_matrix.csv",
        index_col=[0, 1, 2, 3, 4],
    )
    full_df.index.names = [
        "sex",
        "education",
        "age",
        "partner_state",
        "lead_partner_state",
    ]

    # Set up colors
    colors, line_styles = set_colors()

    # Determine relevant ages
    all_ages = np.arange(specs["start_age"], specs["end_age"])

    # Labels
    all_partner_labels = specs["partner_labels"]
    partner_state_vals = list(range(specs["n_partner_states"]))

    col_count = 0
    fig2, axs2 = plt.subplots(nrows=3, ncols=3, figsize=get_figsize(3, 3))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):

            # Plot transition matrix elements
            for current_partner_state, partner_label in enumerate(all_partner_labels):
                for next_partner_state, next_partner_label in enumerate(
                    all_partner_labels
                ):
                    axs2[current_partner_state, next_partner_state].plot(
                        all_ages,
                        full_df.loc[
                            (
                                sex_label,
                                edu_label,
                                all_ages,
                                partner_label,
                                next_partner_label,
                            )
                        ].values,
                        label=f"{sex_label}; {edu_label}",
                        color=colors[col_count % len(colors)],
                    )

            # Note: Predicted vs empirical shares plotting would require additional data
            # that's currently generated during estimation. This can be added later
            # if the estimation functions are modified to save intermediate results.

            col_count += 1

    axs2[0, 0].legend()

    # Set titles for transition matrix subplots
    for i, from_state in enumerate(all_partner_labels):
        for j, to_state in enumerate(all_partner_labels):
            axs2[i, j].set_title(f"From {from_state} to {to_state}")
            axs2[i, j].set_xlabel("Age")
            axs2[i, j].set_ylabel("Transition Probability")

    fig2.suptitle("Partner State Transition Probabilities by Age")
    plt.tight_layout()

    if save:
        fig2.savefig(
            path_dict["first_step_plots"] + "family_transition_matrices.pdf",
            bbox_inches="tight",
        )
        fig2.savefig(
            path_dict["first_step_plots"] + "family_transition_matrices.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()

    plt.close(fig2)


def plot_children(path_dict, specs, show=False, save=False):
    """Plot the number of children by age.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    specs : dict
        Dictionary containing model specifications
    show : bool, default False
        Whether to display plots
    save : bool, default False
        Whether to save plots to disk
    """
    # Calculate the number of children in the household for each individual conditional
    # on sex, education and age bin.
    df = pd.read_pickle(
        path_dict["first_step_data"] + "partner_transition_estimation_sample.pkl"
    )

    start_age = specs["start_age"]
    end_age = specs["end_age"]
    df = df[df["age"] <= end_age]

    df["has_partner"] = (df["partner_state"] > 0).astype(int)

    # calculate average hours worked by partner by age, sex and education
    cov_list = ["sex", "education", "has_partner", "age"]
    nb_children_data = df.groupby(cov_list)["children"].mean()

    nb_children_est = specs["children_by_state"]
    ages = np.arange(start_age, end_age + 1)

    fig, axs = plt.subplots(ncols=4, figsize=get_figsize(ncols=4))
    i = 0

    colors, _ = set_colors()
    sex_labels = ["Men", "Women"]
    partner_labels = ["Single", "Partnered"]

    for sex, sex_label in enumerate(sex_labels):
        for has_partner, partner_label in enumerate(partner_labels):
            ax = axs[i]
            i += 1
            for edu, edu_label in enumerate(specs["education_labels"]):
                nb_children_data_edu = nb_children_data.loc[
                    (sex, edu, has_partner, slice(None))
                ]
                nb_children_container = pd.Series(data=0, index=ages, dtype=float)
                nb_children_container.update(nb_children_data_edu)

                nb_children_est_edu = nb_children_est[sex, edu, has_partner, :]
                ax.plot(
                    ages,
                    nb_children_container,
                    color=colors[edu],
                    linestyle="--",
                    label=f"Obs. {edu_label}",
                )
                ax.plot(
                    ages,
                    nb_children_est_edu,
                    color=colors[edu],
                    label=f"Est. {edu_label}",
                )

            ax.set_ylim([0, 2.5])
            ax.set_title(f"{sex_label}, {partner_label}")

    axs[0].legend()
    plt.tight_layout()

    if save:
        fig.savefig(path_dict["first_step_plots"] + "children.pdf", bbox_inches="tight")
        fig.savefig(
            path_dict["first_step_plots"] + "children.png", bbox_inches="tight", dpi=300
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_marriage_and_divorce(path_dict, specs, show=False, save=False):
    """Illustrate the marriage and divorce rates by age.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    specs : dict
        Dictionary containing model specifications
    show : bool, default False
        Whether to display plots
    save : bool, default False
        Whether to save plots to disk
    """
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    df = pd.read_pickle(
        path_dict["first_step_data"] + "partner_transition_estimation_sample.pkl"
    )
    grouped_shares = df.groupby(["sex", "education", "age"])[
        "partner_state"
    ].value_counts(normalize=True)
    partner_shares_obs = grouped_shares.loc[
        (slice(None), slice(None), slice(None), slice(None))
    ]

    ages = np.arange(start_age, end_age + 1 - 10)
    initial_dist = np.zeros(specs["n_partner_states"])

    fig, axs = plt.subplots(nrows=2, ncols=specs["n_partner_states"], figsize=get_figsize(2, specs["n_partner_states"]))
    colors, _ = set_colors()

    for partner_state, partner_label in enumerate(specs["partner_labels"]):
        for sex_var, sex_label in enumerate(specs["sex_labels"]):
            ax = axs[sex_var, partner_state]
            for edu, edu_label in enumerate(specs["education_labels"]):
                edu_shares_obs = partner_shares_obs.loc[
                    (sex_var, edu, slice(None), partner_state)
                ]
                # Assign only single and married shares at start
                initial_dist[0] = partner_shares_obs.loc[(sex_var, edu, 30, 0)]
                initial_dist[1] = 1 - initial_dist[0]
                shares_over_time = markov_simulator_family(
                    initial_dist,
                    specs["partner_trans_mat"][sex_var, edu, :, :, :],
                    n_periods=len(ages),
                )
                relev_share = shares_over_time[:, partner_state]

                # Use fifty percent as default if not available in the data. Just for plotting
                share_data_container = pd.Series(data=0.0, index=ages, dtype=float)
                share_data_container.update(edu_shares_obs)

                ax.plot(
                    ages,
                    relev_share,
                    color=colors[edu],
                    label=f"Est. {edu_label}",
                )
                ax.plot(
                    ages,
                    share_data_container,
                    color=colors[edu],
                    linestyle="--",
                    label=f"Obs. {edu_label}",
                )
                ax.set_ylim([0, 1])

            ax.set_title(f"{sex_label}; {partner_label}")

    axs[0, 0].legend(loc="upper center")
    plt.tight_layout()

    if save:
        fig.savefig(
            path_dict["first_step_plots"] + "partner_lifecycle.pdf", bbox_inches="tight"
        )
        fig.savefig(
            path_dict["first_step_plots"] + "partner_lifecycle.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def markov_simulator_family(initial_dist, trans_probs, n_periods=None):
    """Simulate a Markov process for family transitions."""
    if n_periods is None:
        n_periods = trans_probs.shape[0]
    else:
        # Check if n_periods is integer
        if not isinstance(n_periods, int):
            raise ValueError("n_periods must be an integer.")

    n_states = initial_dist.shape[0]
    final_dist = np.zeros((n_periods, n_states))
    final_dist[0, :] = initial_dist

    for t in range(n_periods - 1):
        current_dist = final_dist[t, :]
        for state in range(n_states - 1):
            final_dist[t + 1, state] = current_dist @ trans_probs[t, :, state]

        final_dist[t + 1, -1] = 1 - final_dist[t + 1, :-1].sum()

    return final_dist
    plt.close(fig2)


def plot_predicted_vs_empirical_shares(
    path_dict,
    specs,
    predicted_shares_data,
    empirical_shares_data,
    show=False,
    save=False,
):
    """Plot predicted vs empirical partner state shares by age.

    This function is designed to be called from the estimation module
    when the intermediate data is available.
    """

    # Set up colors
    colors, line_styles = set_colors()

    # Determine relevant ages
    all_ages = np.arange(specs["start_age"], specs["end_age"])
    partner_state_vals = list(range(specs["n_partner_states"]))

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=get_figsize(2, 2))
    col_count = 0

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axs[col_count // 2, col_count % 2]

            pred_shares = predicted_shares_data[(sex_label, edu_label)]
            emp_shares = empirical_shares_data[(sex_label, edu_label)]

            for current_partner_state in partner_state_vals:
                # Predicted shares (solid line)
                ax.plot(
                    all_ages,
                    pred_shares.loc[(all_ages, current_partner_state)],
                    label=f"Pred: {current_partner_state}",
                    color=colors[current_partner_state % len(colors)],
                    linestyle="-",
                )
                # Empirical shares (dashed line)
                ax.plot(
                    all_ages,
                    emp_shares.loc[(all_ages, current_partner_state)],
                    label=f"Emp: {current_partner_state}",
                    color=colors[current_partner_state % len(colors)],
                    linestyle="--",
                )

            ax.set_title(f"{sex_label}, {edu_label}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Share")
            ax.legend()
            col_count += 1

    fig.suptitle("Predicted vs Empirical Partner State Shares")
    plt.tight_layout()

    if save:
        fig.savefig(
            path_dict["first_step_plots"] + "family_predicted_vs_empirical.pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            path_dict["first_step_plots"] + "family_predicted_vs_empirical.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()

    plt.close(fig)


def plot_children(path_dict, specs, show=False, save=False):
    """Plot the number of children by age.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    specs : dict
        Dictionary containing model specifications
    show : bool, default False
        Whether to display plots
    save : bool, default False
        Whether to save plots to disk
    """
    # Calculate the number of children in the household for each individual conditional
    # on sex, education and age bin.
    df = pd.read_pickle(
        path_dict["first_step_data"] + "partner_transition_estimation_sample.pkl"
    )

    start_age = specs["start_age"]
    end_age = specs["end_age"]
    df = df[df["age"] <= end_age]

    df["has_partner"] = (df["partner_state"] > 0).astype(int)

    # calculate average hours worked by partner by age, sex and education
    cov_list = ["sex", "education", "has_partner", "age"]
    nb_children_data = df.groupby(cov_list)["children"].mean()

    nb_children_est = specs["children_by_state"]
    ages = np.arange(start_age, end_age + 1)

    fig, axs = plt.subplots(ncols=4, figsize=get_figsize(ncols=4))
    i = 0

    colors, _ = set_colors()
    sex_labels = ["Men", "Women"]
    partner_labels = ["Single", "Partnered"]

    for sex, sex_label in enumerate(sex_labels):
        for has_partner, partner_label in enumerate(partner_labels):
            ax = axs[i]
            i += 1
            for edu, edu_label in enumerate(specs["education_labels"]):
                nb_children_data_edu = nb_children_data.loc[
                    (sex, edu, has_partner, slice(None))
                ]
                nb_children_container = pd.Series(data=0, index=ages, dtype=float)
                nb_children_container.update(nb_children_data_edu)

                nb_children_est_edu = nb_children_est[sex, edu, has_partner, :]
                ax.plot(
                    ages,
                    nb_children_container,
                    color=colors[edu],
                    linestyle="--",
                    label=f"Obs. {edu_label}",
                )
                ax.plot(
                    ages,
                    nb_children_est_edu,
                    color=colors[edu],
                    label=f"Est. {edu_label}",
                )

            ax.set_ylim([0, 2.5])
            ax.set_title(f"{sex_label}, {partner_label}")

    axs[0].legend()
    plt.tight_layout()

    if save:
        fig.savefig(path_dict["first_step_plots"] + "children.pdf", bbox_inches="tight")
        fig.savefig(
            path_dict["first_step_plots"] + "children.png", bbox_inches="tight", dpi=300
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_marriage_and_divorce(path_dict, specs, show=False, save=False):
    """Illustrate the marriage and divorce rates by age.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    specs : dict
        Dictionary containing model specifications
    show : bool, default False
        Whether to display plots
    save : bool, default False
        Whether to save plots to disk
    """
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    df = pd.read_pickle(
        path_dict["first_step_data"] + "partner_transition_estimation_sample.pkl"
    )
    grouped_shares = df.groupby(["sex", "education", "age"])[
        "partner_state"
    ].value_counts(normalize=True)
    partner_shares_obs = grouped_shares.loc[
        (slice(None), slice(None), slice(None), slice(None))
    ]

    ages = np.arange(start_age, end_age + 1 - 10)
    initial_dist = np.zeros(specs["n_partner_states"])

    fig, axs = plt.subplots(nrows=2, ncols=specs["n_partner_states"], figsize=get_figsize(2, specs["n_partner_states"]))
    colors, _ = set_colors()

    for partner_state, partner_label in enumerate(specs["partner_labels"]):
        for sex_var, sex_label in enumerate(specs["sex_labels"]):
            ax = axs[sex_var, partner_state]
            for edu, edu_label in enumerate(specs["education_labels"]):
                edu_shares_obs = partner_shares_obs.loc[
                    (sex_var, edu, slice(None), partner_state)
                ]
                # Assign only single and married shares at start
                initial_dist[0] = partner_shares_obs.loc[(sex_var, edu, 30, 0)]
                initial_dist[1] = 1 - initial_dist[0]
                shares_over_time = markov_simulator_family(
                    initial_dist,
                    specs["partner_trans_mat"][sex_var, edu, :, :, :],
                    n_periods=len(ages),
                )
                relev_share = shares_over_time[:, partner_state]

                # Use fifty percent as default if not available in the data. Just for plotting
                share_data_container = pd.Series(data=0.0, index=ages, dtype=float)
                share_data_container.update(edu_shares_obs)

                ax.plot(
                    ages,
                    relev_share,
                    color=colors[edu],
                    label=f"Est. {edu_label}",
                )
                ax.plot(
                    ages,
                    share_data_container,
                    color=colors[edu],
                    linestyle="--",
                    label=f"Obs. {edu_label}",
                )
                ax.set_ylim([0, 1])

            ax.set_title(f"{sex_label}; {partner_label}")

    axs[0, 0].legend(loc="upper center")
    plt.tight_layout()

    if save:
        fig.savefig(
            path_dict["first_step_plots"] + "partner_lifecycle.pdf", bbox_inches="tight"
        )
        fig.savefig(
            path_dict["first_step_plots"] + "partner_lifecycle.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def markov_simulator_family(initial_dist, trans_probs, n_periods=None):
    """Simulate a Markov process for family transitions."""
    if n_periods is None:
        n_periods = trans_probs.shape[0]
    else:
        # Check if n_periods is integer
        if not isinstance(n_periods, int):
            raise ValueError("n_periods must be an integer.")

    n_states = initial_dist.shape[0]
    final_dist = np.zeros((n_periods, n_states))
    final_dist[0, :] = initial_dist

    for t in range(n_periods - 1):
        current_dist = final_dist[t, :]
        for state in range(n_states - 1):
            final_dist[t + 1, state] = current_dist @ trans_probs[t, :, state]

        final_dist[t + 1, -1] = 1 - final_dist[t + 1, :-1].sum()

    return final_dist
