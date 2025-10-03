# Description: This file contains plotting functions for health state estimation results.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from set_styles import set_colors, get_figsize


def plot_healthy_unhealthy(path_dict, specs, show=False, save=False):
    """Illustrate the health rates by age (actual vs. estimated by markov chain).
    
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
    # Load the data and define age range
    start_age = specs["start_age"]
    end_age = specs["end_age"]

    max_age_est_physical = 90
    max_period_physical = max_age_est_physical - start_age
    est_ages = np.arange(start_age, end_age + 1)

    # Load the health transition sample
    df = pd.read_pickle(
        path_dict["first_step_data"] + "health_transition_estimation_sample.pkl"
    )

    good_health_var = specs["good_health_var"]
    bad_health_var = specs["bad_health_var"]
    death_health_var = specs["death_health_var"]

    # Calculate the smoothed shares for healthy individuals
    edu_shares_data_healthy = (
        df.groupby(["sex", "education", "age"])["health"]
        .value_counts(normalize=True)
        .loc[slice(None), slice(None), slice(start_age, end_age + 1), good_health_var]
    )
    edu_shares_data_healthy.index = edu_shares_data_healthy.index.droplevel("health")
    full_index = pd.MultiIndex.from_product(
        [range(specs["n_sexes"]), range(specs["n_education_types"]), est_ages],
        names=["sex", "education", "age"],
    )
    edu_shares_healthy = pd.DataFrame(
        index=full_index, columns=["proportion"], data=0.0
    )
    edu_shares_healthy.update(edu_shares_data_healthy)

    # Initialize the distribution
    initial_dist = np.zeros(specs["n_all_health_states"])

    # Create the plot
    fig, axs = plt.subplots(ncols=2, figsize=get_figsize(ncols=2))
    colors, _ = set_colors()
    
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axs[sex_var]
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            # Set the initial distribution for the Markov simulation
            # and assume nobody is dead
            initial_dist[good_health_var] = edu_shares_healthy.loc[
                (sex_var, edu_var, start_age), "proportion"
            ]
            initial_dist[bad_health_var] = 1 - initial_dist[good_health_var]

            # Simulate the Markov process and get health probabilities
            shares_over_time = markov_simulator(
                initial_dist, specs["health_trans_mat"][sex_var, edu_var, :, :, :]
            )
            # Construct health probabilities if alive (Use death probs as column vector)
            alive_share = 1 - shares_over_time[:, death_health_var]
            healthy_share_type_est = shares_over_time[:, good_health_var] / alive_share
            healthy_share_type_data = edu_shares_healthy.loc[
                (sex_var, edu_var, slice(None))
            ].values

            # Plot the estimates and the data
            ax.plot(
                est_ages[:max_period_physical],
                healthy_share_type_est[:max_period_physical],
                color=colors[edu_var],
                label=f"Est. {edu_label}",
            )
            ax.plot(
                est_ages[:max_period_physical],
                healthy_share_type_data[:max_period_physical],
                color=colors[edu_var],
                linestyle="--",
                label=f"Obs. {edu_label}",
            )

            # Adjust the x-axis ticks and labels
            x_ticks = np.arange(start_age, max_age_est_physical + 1, 10)
            ax.set_xticks(x_ticks)
            # Set font size for x-axis labels
            ax.set_xticklabels(x_ticks, fontsize=12)

        # Set y-axis limits and ticks
        ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        # Set yticks labels and fontsize
        ax.set_yticklabels([f"{i:.0%}" for i in np.arange(0.0, 1.1, 0.1)], fontsize=12)

        # Add title and legend
        ax.set_title(f"{sex_label}", fontsize=14)
        ax.set_xlabel("Age", fontsize=12)
        ax.set_ylabel("Probability of being Healthy", fontsize=12)
        
    axs[0].legend()
    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["first_step_plots"] + "health_states.pdf", bbox_inches="tight")
        fig.savefig(path_dict["first_step_plots"] + "health_states.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_health_transition_prob(path_dict, specs, show=False, save=False):
    """Plot health transition probabilities by age.
    
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
    fig, axs = plt.subplots(2, 1, figsize=get_figsize(2, 1), sharex=True)
    colors, _ = set_colors()

    for health_var in specs["observed_health_vars"]:
        color_id = 0
        ax = axs[health_var]
        for sex_var, sex_label in enumerate(specs["sex_labels"]):
            for edu_var, edu_label in enumerate(specs["education_labels"]):
                health_var_to = 1 - health_var
                ax.plot(
                    periods,
                    trans_probs[sex_var, edu_var, :, health_var, health_var_to],
                    color=colors[color_id],
                    label=f"{sex_label}; {edu_label}",
                )
                # Shock is reverse from which is baseline health
                label = specs["health_labels"][health_var_to]
                ax.set_title(
                    f"Probability of {label} Shock (kernel-smoothed), bw={bandwidth}",
                    fontsize=14,
                )
                ax.set_ylabel("Probability", fontsize=12)
                ax.legend(loc="upper right")
                ax.set_ylim(0, 0.4)
                ax.set_yticks(
                    [i * 0.05 for i in range(9)]
                )  # Y-axis ticks from 0 to 0.4 with steps of 0.05
                # Set the x-axis ticks and labels
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(age_ticks)
                ax.grid(False)
                color_id += 1
                
    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["first_step_plots"] + "health_transition_probs.pdf", bbox_inches="tight")
        fig.savefig(path_dict["first_step_plots"] + "health_transition_probs.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)


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