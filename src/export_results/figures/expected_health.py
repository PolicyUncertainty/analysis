 
 
 
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
    edu_shares_healthy = df.groupby(["education", "age"])[
        "health_state"
    ].mean().rolling(window=specs["health_smoothing_bandwidth"], center=True).mean().loc[slice(None), slice(specs["start_age"], specs["end_age"])]

    ages = np.arange(start_age, end_age + 1)
    initial_dist = np.zeros(specs["n_health_states"])

    fig, ax = plt.subplots(figsize=(12, 8))
    for edu, edu_label in enumerate(specs["education_labels"]):
        initial_dist[1] = edu_shares_healthy.loc[(edu, start_age)]
        initial_dist[0] = 1 - initial_dist[1]
        shares_over_time = markov_simulator(
            initial_dist, specs["health_trans_mat"][edu, :, :, :]
        )
        health_prob_edu_est = shares_over_time[:, 1]
        health_prob_edu_data = (
            edu_shares_healthy.loc[(edu, slice(None))].values
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



def plot_health_transition_prob(specs):
    # Create the figures for the health transition probabilities
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Load the transition probabilities
    trans_probs =  specs["health_trans_mat"]

    # Define the age range
    peroids = range(specs["n_periods"])

    # Define the colors
    colors = ["tab:blue", "tab:orange"]

    # Define the bandwidth for the kernel density estimation
    bandwidth = specs["health_smoothing_bandwidth"]

    # Panel (a): Probability of good health shock
    axs[0].plot(peroids, trans_probs[0,:,0,1], color=colors[1], label="Low education (smoothed)")
    axs[0].scatter(peroids, trans_probs[0,:,0,1], color=colors[1], alpha=0.65, label="Low education")
    axs[0].plot(peroids, trans_probs[1,:,0,1], color=colors[0], label="High education (smoothed)")
    axs[0].scatter(peroids, trans_probs[1,:,0,1], color=colors[0], alpha=0.65, label="High education")
    axs[0].set_title(f"Probability of Good Health Shock (kernel-smoothed), bw={bandwidth}", fontsize=14)
    axs[0].set_ylabel("Probability", fontsize=12)
    axs[0].set_xlabel("Age (years)", fontsize=12)
    axs[0].legend(loc="upper right")
    # axs[0].set_ylim(0, 0.6)
    # axs[0].set_yticks(np.arange(0, 0.7, 0.1))
    axs[0].grid(False)

    # Panel (b): Probability of bad health shock
    axs[1].plot(peroids, trans_probs[0,:,1,0], color=colors[1], label="Low education (smoothed)")
    axs[1].scatter(peroids, trans_probs[0,:,1,0], color=colors[1], alpha=0.65, label="Low education")
    axs[1].plot(peroids, trans_probs[1,:,1,0], color=colors[0], label="High education (smoothed)")
    axs[1].scatter(peroids, trans_probs[1,:,1,0], color=colors[0], alpha=0.65, label="High education")
    axs[1].set_title(f"Probability of Bad Health Shock (kernel-smoothed), bw={bandwidth}", fontsize=14)
    axs[1].set_xlabel("Age (years)", fontsize=12)
    axs[1].set_ylabel("Probability", fontsize=12)
    axs[1].legend(loc="upper right")
    # axs[1].set_ylim(0, 0.6)
    # axs[1].set_yticks(np.arange(0, 0.7, 0.1))
    axs[1].grid(False)
