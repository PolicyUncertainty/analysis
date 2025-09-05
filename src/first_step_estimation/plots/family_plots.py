# Description: This file contains plotting functions for family transition estimation results.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import set_colors


def plot_family_transition_results(path_dict, specs, show=False, save=False):
    """Plot family transition estimation results including transition matrices and predicted vs empirical shares."""
    
    # Load the transition matrix results
    full_df = pd.read_csv(
        path_dict["est_results"] + "partner_transition_matrix.csv", 
        index_col=[0, 1, 2, 3, 4]
    )
    full_df.index.names = ["sex", "education", "age", "partner_state", "lead_partner_state"]
    
    # Set up colors
    colors, line_styles = set_colors()
    
    # Determine relevant ages
    all_ages = np.arange(specs["start_age"], specs["end_age"])
    
    # Labels
    all_partner_labels = specs["partner_labels"]
    partner_state_vals = list(range(specs["n_partner_states"]))
    
    col_count = 0
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig2, axs2 = plt.subplots(nrows=3, ncols=3)
    
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            
            # Plot transition matrix elements
            for current_partner_state, partner_label in enumerate(all_partner_labels):
                for next_partner_state, next_partner_label in enumerate(all_partner_labels):
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
        fig2.savefig(path_dict["first_step_plots"] + "family_transition_matrices.pdf", 
                    bbox_inches="tight")
        fig2.savefig(path_dict["first_step_plots"] + "family_transition_matrices.png", 
                    bbox_inches="tight", dpi=300)
    
    if show:
        plt.show()
    
    plt.close(fig)
    plt.close(fig2)


def plot_predicted_vs_empirical_shares(path_dict, specs, predicted_shares_data, empirical_shares_data, show=False, save=False):
    """Plot predicted vs empirical partner state shares by age.
    
    This function is designed to be called from the estimation module
    when the intermediate data is available.
    """
    
    # Set up colors
    colors, line_styles = set_colors()
    
    # Determine relevant ages
    all_ages = np.arange(specs["start_age"], specs["end_age"])
    partner_state_vals = list(range(specs["n_partner_states"]))
    
    fig, axs = plt.subplots(nrows=2, ncols=2)
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
                    linestyle="-"
                )
                # Empirical shares (dashed line)
                ax.plot(
                    all_ages,
                    emp_shares.loc[(all_ages, current_partner_state)],
                    label=f"Emp: {current_partner_state}",
                    color=colors[current_partner_state % len(colors)],
                    linestyle="--"
                )
            
            ax.set_title(f"{sex_label}, {edu_label}")
            ax.set_xlabel("Age")
            ax.set_ylabel("Share")
            ax.legend()
            col_count += 1
    
    fig.suptitle("Predicted vs Empirical Partner State Shares")
    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["first_step_plots"] + "family_predicted_vs_empirical.pdf", 
                   bbox_inches="tight")
        fig.savefig(path_dict["first_step_plots"] + "family_predicted_vs_empirical.png", 
                   bbox_inches="tight", dpi=300)
    
    if show:
        plt.show()
    
    plt.close(fig)