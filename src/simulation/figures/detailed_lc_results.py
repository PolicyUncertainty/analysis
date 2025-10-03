import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import get_figsize, set_colors


def plot_detailed_lifecycle_results(
    df_baseline,
    path_dict,
    specs,
    model_name,
    df_comparison=None,
    comparison_name=None,
    show=False,
    save=True,
    end_age=80,
):
    """
    Plot detailed lifecycle results by demographic groups.

    Parameters:
    -----------
    df_results_path : str
        Path to CSV file with multi-index DataFrame from calc_life_cycle_detailed
    path_dict : dict
        Path dictionary for saving plots
    specs : dict
        Model specifications with labels
    subfolder : str, optionalsubfolder
        Subfolder within simulation_plots to save plots
    df_results_comparison_path : str, optional
        Path to comparison CSV file to overlay with dotted lines
    comparison_name : str, optional
        Name for comparison data (default: "comparison")
    show : bool
        Whether to display plots
    save : bool
        Whether to save plots
    """

    comp_name = comparison_name or "comparison"

    plot_ages = np.arange(specs.get("start_age", 30), end_age, dtype=int)

    plot_dir = path_dict["simulation_plots"] + f"{model_name}/baseline/"
    # Check if directory exists, if not create it
    if save and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    colors, _ = set_colors()

    # Define outcome variables and their display names
    outcomes = {
        "choice_0_rate": "Retirement Rate",
        "choice_1_rate": "Unemployment Rate",
        "choice_2_rate": "Part-time Work Rate",
        "choice_3_rate": "Full-time Work Rate",
        "savings_rate": "Savings Rate",
        "avg_wealth": "Average Wealth",
        "consumption": "Consumption",
        "gross_own_income": "Gross Own Income",
        "net_hh_income": "HH Net Income",
    }

    # Define group types to plot (exclude aggregate)
    group_types = ["sex", "education", "initial_informed", "health", "partner_state"]

    # Get group labels from specs
    group_labels = {
        "sex": specs.get("sex_labels", ["Male", "Female"]),
        "education": specs.get("education_labels", ["Low Education", "High Education"]),
        "informed": ["Uninformed", "Informed"],
        "health": ["Good Health", "Bad Health", "Disabled", "Dead"],
        "partner_state": ["Single", "Partnered", "Retired Partner"],
    }

    for group_type in group_types:
        # Skip if group not in data
        if group_type not in df_baseline.index.get_level_values("group_type"):
            continue

        # Skip if group not in comparison data (if provided)
        if (
            df_comparison is not None
            and group_type not in df_comparison.index.get_level_values("group_type")
        ):
            continue

        # Create figure with subplots for each outcome
        fig, axes = plt.subplots(3, 3, figsize=get_figsize(3, 3))
        axes = axes.flatten()

        group_data = df_baseline.loc[group_type]
        group_values = group_data.index.get_level_values("group_value").unique()

        # Get comparison data if available
        group_data_comp = (
            df_comparison.loc[group_type] if df_comparison is not None else None
        )

        for i, (outcome_var, outcome_name) in enumerate(outcomes.items()):
            ax = axes[i]

            # Plot line for each group value - main data (solid lines)
            for j, group_val in enumerate(sorted(group_values)):
                if group_val in group_data.index.get_level_values("group_value"):
                    data = group_data.loc[group_val][outcome_var]

                    # Convert group_val to int for indexing (CSV loading makes it a string)
                    try:
                        group_val_int = int(float(group_val))
                        if group_val_int < len(group_labels[group_type]):
                            label = group_labels[group_type][group_val_int]
                        else:
                            label = f"{group_type}={group_val}"
                    except (ValueError, TypeError):
                        label = f"{group_type}={group_val}"

                    ax.plot(
                        plot_ages,
                        data.reindex(plot_ages).values,
                        color=colors[j % len(colors)],
                        label=label,
                        linewidth=2,
                        linestyle="-",
                    )

                    # Plot comparison data (dotted lines) if available
                    if (
                        group_data_comp is not None
                        and group_val
                        in group_data_comp.index.get_level_values("group_value")
                    ):
                        data_comp = group_data_comp.loc[group_val][outcome_var]
                        ax.plot(
                            plot_ages,
                            data_comp.reindex(plot_ages).values,
                            color=colors[j % len(colors)],
                            label=f"{label} ({comp_name})",
                            linewidth=2,
                            linestyle="--",
                        )

            ax.set_xlabel("Age")
            ax.set_ylabel(outcome_name)
            ax.set_title(outcome_name)
            ax.legend()
            # ax.grid(True, alpha=0.3)

        # plt.suptitle(f'Lifecycle Profiles by {group_type.title()}', fontsize=16)
        plt.tight_layout()

        if save:
            filename = f"lifecycle_profiles_by_{group_type}"
            fig.savefig(plot_dir + f"{filename}.pdf", bbox_inches="tight")
            fig.savefig(plot_dir + f"{filename}.png", bbox_inches="tight", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)
