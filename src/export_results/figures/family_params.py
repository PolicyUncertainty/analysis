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
                    (sex, edu, has_partner, slice(None))
                ]
                nb_children_container = pd.Series(data=0, index=ages, dtype=float)
                nb_children_container.update(nb_children_data_edu)

                nb_children_est_edu = nb_children_est[sex, edu, has_partner, :]
                ax.plot(ages, nb_children_container, label=f"edu {edu}")
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
    single_shares = grouped_shares.loc[(slice(None), slice(None), slice(None), 0)]

    ages = np.arange(start_age, end_age + 1)
    initial_dist = np.zeros(specs["n_partner_states"])

    fig, axs = plt.subplots(nrows=2, figsize=(12, 8))
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axs[sex_var]
        for edu, edu_label in enumerate(specs["education_labels"]):
            edu_shares_single = single_shares.loc[(sex_var, edu, slice(None))]
            initial_dist[0] = edu_shares_single.loc[start_age]
            initial_dist[1] = 1 - initial_dist[0]
            shares_over_time = markov_simulator(
                initial_dist, specs["partner_trans_mat"][sex_var, edu, :, :, :]
            )
            marriage_prob_edu_est = 1 - shares_over_time[:, 0]

            # Use fifty percent as default if not available in the data. Just for plotting
            married_shares_data_containier = pd.Series(
                data=0.5, index=ages, dtype=float
            )
            married_shares_data_containier.update(1 - edu_shares_single)

            ax.plot(ages, marriage_prob_edu_est, label=f"edu {edu_label} est")
            ax.plot(
                ages,
                married_shares_data_containier,
                linestyle="--",
                label=f"edu {edu_label} data",
            )

        ax.set_title(f"Marriage shares {sex_label}")
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
