import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define more visually appealing red and blue colors
colors = {0: "#D72638", 1: "#1E90FF"}  # Red: #D72638, Blue: #1E90FF


def plot_healthy_unhealthy(paths_dict, specs):
    """Illustrate the health rates by age.

    (actual vs. estimated by markov chain)

    """

    # Load the data and define age range
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    ages = np.arange(start_age, end_age + 1)

    # Load the health transition sample
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "health_transition_estimation_sample.pkl"
    )

    # Calculate the smoothed shares for healthy individuals
    edu_shares_healthy = (
        df.groupby(["education", "age"])["health"]
        .mean()
        .loc[slice(None), slice(start_age, end_age)]
    )

    alive_healths = np.where(np.array(specs["health_labels"]) != "Death")[0]
    n_alive_health_states = len(alive_health_states)

    # Initialize the distribution
    initial_dist = np.zeros(n_alive_health_states)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for edu, edu_label in enumerate(specs["education_labels"]):
        # Set the initial distribution for the Markov simulation
        initial_dist[1] = edu_shares_healthy.loc[(edu, start_age)]
        initial_dist[0] = 1 - initial_dist[1]

        # Simulate the Markov process and get health probabilities
        shares_over_time = markov_simulator(
            initial_dist, specs["health_trans_mat"][edu, :, :, :]
        )
        health_prob_edu_est = shares_over_time[:, 1]
        health_prob_edu_data = edu_shares_healthy.loc[(edu, slice(None))].values

        # Plot the estimates and the data
        ax.plot(
            ages,
            health_prob_edu_est,
            color=colors[edu],
            label=f"{edu_label} MC-Estimate",
        )
        ax.plot(
            ages,
            health_prob_edu_data,
            linestyle="--",
            color=colors[edu],
            label=f"{edu_label} Data, RM w. BW={specs['health_smoothing_bandwidth']}",
        )
        ax.scatter(ages, health_prob_edu_data, color=colors[edu], alpha=0.5, s=8)

    # Adjust the x-axis ticks and labels
    x_ticks = np.arange(start_age, end_age + 1, 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    # Set y-axis limits and ticks
    ax.set_ylim(0.4, 1)  # Set y-axis limits from 0 to 1
    ax.set_yticks(np.arange(0.4, 1.1, 0.1))  # Set y-axis ticks at intervals of 0.1

    # Add title and legend
    ax.set_title("Health shares", fontsize=14)
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Probability of being Healthy", fontsize=12)
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


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
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Load the transition probabilities
    trans_probs = specs["health_trans_mat"]

    # Define the age range and periods
    n_periods = specs["n_periods"]
    start_age = specs["start_age"]
    periods = range(n_periods)

    # Calculate the tick labels for the x-axis
    age_ticks = [start_age + p for p in range(0, n_periods, 10)]
    tick_positions = list(range(0, n_periods, 10))

    # Define the bandwidth for the kernel density estimation
    bandwidth = specs["health_smoothing_bandwidth"]

    # Panel (a): Probability of good health shock
    axs[0].plot(
        periods,
        trans_probs[0, :, 0, 1],
        color=colors[1],
        label="Low education (smoothed)",
    )
    axs[0].scatter(
        periods,
        trans_probs[0, :, 0, 1],
        color=colors[1],
        alpha=0.65,
        s=8,
        label="Low education",
    )
    axs[0].plot(
        periods,
        trans_probs[1, :, 0, 1],
        color=colors[0],
        label="High education (smoothed)",
    )
    axs[0].scatter(
        periods,
        trans_probs[1, :, 0, 1],
        color=colors[0],
        alpha=0.65,
        s=8,
        label="High education",
    )
    axs[0].set_title(
        f"Probability of Good Health Shock (kernel-smoothed), bw={bandwidth}",
        fontsize=14,
    )
    axs[0].set_ylabel("Probability", fontsize=12)
    axs[0].legend(loc="upper right")
    axs[0].set_ylim(0, 0.4)
    axs[0].set_yticks(
        [i * 0.05 for i in range(9)]
    )  # Y-axis ticks from 0 to 0.4 with steps of 0.05
    axs[0].grid(False)

    # Panel (b): Probability of bad health shock
    axs[1].plot(
        periods,
        trans_probs[0, :, 1, 0],
        color=colors[1],
        label="Low education (smoothed)",
    )
    axs[1].scatter(
        periods,
        trans_probs[0, :, 1, 0],
        color=colors[1],
        alpha=0.65,
        s=8,
        label="Low education",
    )
    axs[1].plot(
        periods,
        trans_probs[1, :, 1, 0],
        color=colors[0],
        label="High education (smoothed)",
    )
    axs[1].scatter(
        periods,
        trans_probs[1, :, 1, 0],
        color=colors[0],
        alpha=0.65,
        s=8,
        label="High education",
    )
    axs[1].set_title(
        f"Probability of Bad Health Shock (kernel-smoothed), bw={bandwidth}",
        fontsize=14,
    )
    axs[1].set_xlabel("Age", fontsize=12)
    axs[1].set_ylabel("Probability", fontsize=12)
    axs[1].set_ylim(0, 0.4)
    axs[1].set_yticks(
        [i * 0.05 for i in range(9)]
    )  # Y-axis ticks from 0 to 0.4 with steps of 0.05
    axs[1].legend(loc="upper right")
    axs[1].grid(False)

    # Set the x-axis ticks and labels
    axs[1].set_xticks(tick_positions)
    axs[1].set_xticklabels(age_ticks)
