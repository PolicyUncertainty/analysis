import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
                    (sex, edu, has_partner, slice(start_age, end_age))
                ].values
                nb_children_est_edu = nb_children_est[sex, edu, has_partner, :]
                ax.plot(ages, nb_children_data_edu, label=f"edu {edu}")
                ax.plot(
                    ages, nb_children_est_edu, linestyle="--", label=f"edu {edu} est"
                )

            ax.set_ylim([0, 2.5])
            ax.set_title(f"{sex_label}, {partner_label}")
            ax.legend()

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
    edu_shares_single = grouped_shares.loc[
        (0, slice(None), slice(start_age, end_age), 0)
    ]

    ages = np.arange(start_age, end_age + 1)
    initial_dist = np.zeros(specs["n_partner_states"])

    fig, ax = plt.subplots(figsize=(12, 8))
    for edu, edu_label in enumerate(specs["education_labels"]):
        initial_dist[0] = edu_shares_single.loc[(0, edu, start_age, 0)]
        initial_dist[1] = 1 - initial_dist[0]
        shares_over_time = markov_simulator(
            initial_dist, specs["partner_trans_mat"][edu, :, :, :]
        )
        marriage_prob_edu_est = 1 - shares_over_time[:, 0]
        marriage_prob_edu_data = (
            1 - edu_shares_single.loc[(0, edu, slice(None), 0)].values
        )

        ax.plot(ages, marriage_prob_edu_est, label=f"edu {edu_label} est")
        ax.plot(
            ages, marriage_prob_edu_data, linestyle="--", label=f"edu {edu_label} data"
        )

    ax.set_title("Marriage shares")
    ax.legend()


def markov_simulator(initial_dist, trans_probs):
    """Simulate a Markov process."""
    n_periods = trans_probs.shape[0]
    n_states = initial_dist.shape[0]
    final_dist = np.zeros((n_periods, n_states))
    final_dist[0, :] = initial_dist

    for t in range(n_periods - 1):
        current_dist = final_dist[t, :]
        for state in range(n_states - 1):
            final_dist[t + 1, state] = current_dist @ trans_probs[t, :, state]

        final_dist[t + 1, -1] = 1 - final_dist[t + 1, :-1].sum()

    return final_dist
