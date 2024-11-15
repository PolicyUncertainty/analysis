 
 
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_healthy_unhealthy(paths_dict, specs):
    """Illustrate the health rates by age. (actual vs. estimated by markov chain)"""

    start_age = specs["start_age"]
    end_age = specs["end_age"]
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "health_transition_estimation_sample.pkl"
    )
    grouped_shares = df.groupby(["education", "age"])[
        "health_state"
    ].value_counts(normalize=True)
    edu_shares_healthy = grouped_shares.loc[
        (slice(None), slice(start_age, end_age), 0)
    ]

    ages = np.arange(start_age, end_age + 1)
    initial_dist = np.zeros(specs["n_health_states"])

    fig, ax = plt.subplots(figsize=(12, 8))
    for edu, edu_label in enumerate(specs["education_labels"]):
        initial_dist[0] = edu_shares_healthy.loc[(edu, start_age, 0)]
        initial_dist[1] = 1 - initial_dist[0]
        shares_over_time = markov_simulator(
            initial_dist, specs["health_trans_mat"][edu, :, :, :]
        )
        health_prob_edu_est = 1 - shares_over_time[:, 0]
        health_prob_edu_data = (
            1 - edu_shares_healthy.loc[(edu, slice(None), 0)].values
        )

        ax.plot(ages, health_prob_edu_est, label=f"edu {edu_label} est")
        ax.plot(
            ages, health_prob_edu_data, linestyle="--", label=f"edu {edu_label} data"
        )

    ax.set_title("Health shares")
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



def plot_health_transition_prob(paths_dict, specs):
    # Create the figures for the health transition probabilities
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Panel (a): Probability of good health shock
    axs[0].plot(ages, hbg_l, color=colors[1], label="Low education (smoothed)")
    axs[0].scatter(ages, hbg_l, color=colors[1], alpha=0.65, label="Low education")
    axs[0].plot(ages, hbg_h, color=colors[0], label="High education (smoothed)")
    axs[0].scatter(ages, hbg_h, color=colors[0], alpha=0.65, label="High education")
    axs[0].set_title(f"Probability of Good Health Shock (kernel-smoothed), bw={bandwidth}", fontsize=14)
    axs[0].set_ylabel("Probability", fontsize=12)
    axs[0].set_xlabel("Age (years)", fontsize=12)
    axs[0].legend(loc="upper right")
    axs[0].set_ylim(0, 0.6)
    axs[0].set_yticks(np.arange(0, 0.7, 0.1))
    axs[0].set_xticks(np.arange(30, 90, 10))
    axs[0].grid(False)

    # Panel (b): Probability of bad health shock
    axs[1].plot(ages, hgb_l, color=colors[1], label="Low education (smoothed)")
    axs[1].scatter(ages, hgb_l, color=colors[1], alpha=0.65, label="Low education")
    axs[1].plot(ages, hgb_h, color=colors[0], label="High education (smoothed)")
    axs[1].scatter(ages, hgb_h, color=colors[0], alpha=0.65, label="High education")
    axs[1].set_title(f"Probability of Bad Health Shock (kernel-smoothed), bw={bandwidth}", fontsize=14)
    axs[1].set_xlabel("Age (years)", fontsize=12)
    axs[1].set_ylabel("Probability", fontsize=12)
    axs[1].legend(loc="upper right")
    axs[1].set_ylim(0, 0.6)
    axs[1].set_yticks(np.arange(0, 0.7, 0.1))
    axs[1].set_xticks(np.arange(30, 90, 10))
    axs[1].grid(False)

    # Display the plots
    plt.show()