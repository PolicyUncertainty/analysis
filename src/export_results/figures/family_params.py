# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# THIS IS THE LEGACY VERSION - DELETE SOON!
# NEW HOME: src/first_step_estimation/plots/family_plots.py
# FUNCTIONS: plot_children() and plot_marriage_and_divorce() are now combined in family_plots.py
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from export_results.figures.color_map import JET_COLOR_MAP


def plot_children(paths_dict, specs):
    """Plot the number of children by age."""

    """Calculate the number of children in the household for each individual conditional
    on sex, education and age bin."""
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_transition_estimation_sample.pkl"
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

    fig, axs = plt.subplots(ncols=4, figsize=(12, 8))
    i = 0

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
                    color=JET_COLOR_MAP[edu],
                    linestyle="--",
                    label=f"Obs. {edu_label}",
                )
                ax.plot(
                    ages,
                    nb_children_est_edu,
                    color=JET_COLOR_MAP[edu],
                    label=f"Est. {edu_label}",
                )

            ax.set_ylim([0, 2.5])
            ax.set_title(f"{sex_label}, {partner_label}")
        axs[0].legend()
        fig.savefig(paths_dict["plots"] + "children.png")

    # calculate avewealth_shock_scalerage hours worked by partner by


def plot_marriage_and_divorce(paths_dict, specs):
    """Illustrate the marriage and divorce rates by age."""

    start_age = specs["start_age"]
    end_age = specs["end_age"]
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_transition_estimation_sample.pkl"
    )
    grouped_shares = df.groupby(["sex", "education", "age"])[
        "partner_state"
    ].value_counts(normalize=True)
    partner_shares_obs = grouped_shares.loc[
        (slice(None), slice(None), slice(None), slice(None))
    ]

    ages = np.arange(start_age, end_age + 1 - 10)
    initial_dist = np.zeros(specs["n_partner_states"])

    fig, axs = plt.subplots(nrows=2, ncols=specs["n_partner_states"], figsize=(12, 8))
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
                shares_over_time = markov_simulator(
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
                    color=JET_COLOR_MAP[edu],
                    label=f"Est. {edu_label}",
                )
                ax.plot(
                    ages,
                    share_data_container,
                    color=JET_COLOR_MAP[edu],
                    linestyle="--",
                    label=f"Obs. {edu_label}",
                )
                ax.set_ylim([0, 1])

            ax.set_title(f"{sex_label}; {partner_label}")
    axs[0, 0].legend(loc="upper center")
    fig.savefig(paths_dict["plots"] + "partner.png")


def markov_simulator(initial_dist, trans_probs, n_periods=None):
    """Simulate a Markov process."""
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
