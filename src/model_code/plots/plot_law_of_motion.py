import numpy as np
from matplotlib import pyplot as plt

from model_code.state_space.experience import (
    construct_experience_years,
    get_next_period_experience,
    scale_experience_years,
)
from set_styles import get_figsize, set_colors


def plot_ret_experience_multi(path_dict, specs, show=False, save=False):
    """Plot retirement experience law of motion for multiple demographic groups.

    Creates subplots for different combinations of sex, education, and health states,
    with both informed states shown in each subplot.

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
    colors, _ = set_colors()
    periods = np.arange(30, 40)

    # Define the groups to plot
    sex_values = [0, 1]  # 0: Male, 1: Female
    education_values = [0, 1]  # Assuming 0: Low, 1: High education
    informed_values = [0, 1]  # 0: Not informed, 1: Informed
    health_values = [1, 2]  # Health states 1 and 2 only

    # Labels for plotting
    sex_labels = ["Male", "Female"]
    education_labels = ["Low Education", "High Education"]
    informed_labels = ["Not Informed", "Informed"]
    health_labels = ["Health State 1", "Health State 2"]

    # Create one plot for each health state
    for health in health_values:

        fig, axes = plt.subplots(2, 2, figsize=get_figsize(2, 2))
        fig.suptitle(f"{health_labels[health - 1]}", fontweight="bold")

        for sex_idx, sex in enumerate(sex_values):
            for edu_idx, education in enumerate(education_values):

                ax = axes[sex_idx, edu_idx]

                # Plot for both informed states
                for informed_idx, informed in enumerate(informed_values):

                    # Plot different experience levels
                    for i, exp_years in enumerate(np.arange(30, 50, 10)):
                        # Scale the experience for a not retired person last period
                        exp = scale_experience_years(
                            experience_years=exp_years,
                            period=periods - 1,
                            is_retired=np.zeros_like(periods, dtype=bool),
                            model_specs=specs,
                        )

                        exp_next = get_next_period_experience(
                            period=periods,
                            lagged_choice=0,
                            policy_state=8,  # Policy state 8 as in original
                            sex=sex,
                            education=education,
                            experience=exp,
                            informed=informed,
                            health=health,
                            model_specs=specs,
                        )

                        exp_years_next = construct_experience_years(
                            float_experience=exp_next,
                            period=periods,
                            is_retired=np.ones_like(periods, dtype=bool),
                            model_specs=specs,
                        )

                        # Use different line styles for informed status
                        linestyle = "-" if informed == 1 else ":"
                        # alpha = 1.0 if informed == 1 else 0.7

                        ax.plot(
                            periods + 30,
                            exp_years_next,
                            label=f"Exp {exp_years} ({informed_labels[informed]})",
                            color=colors[i % len(colors)],
                            linestyle=linestyle,
                            # alpha=alpha,
                        )

                # Formatting for each subplot
                ax.set_title(f"{sex_labels[sex]}, {education_labels[education]}")
                ax.set_xlabel("Period")
                ax.set_ylabel("Experience Years")
                ax.grid(True, alpha=0.3)

                # Add legend only to the top-right subplot to avoid clutter
                if sex_idx == 0 and edu_idx == 1:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        if save:
            filename_base = f"retirement_experience_law_of_motion_health_{health}"
            fig.savefig(
                path_dict["model_plots"] + f"{filename_base}.pdf", bbox_inches="tight"
            )
            fig.savefig(
                path_dict["model_plots"] + f"{filename_base}.png",
                bbox_inches="tight",
                dpi=300,
            )

        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_ret_experience_combined(path_dict, specs, show=False, save=False):
    """Plot retirement experience law of motion with all groups in a single large grid.

    Creates a comprehensive plot with all combinations of sex, education,
    and health states (1 and 2 only) with both informed states in each subplot.

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
    colors, _ = set_colors()
    periods = np.arange(30, 40)

    # Define the groups to plot
    sex_values = [0, 1]  # 0: Male, 1: Female
    education_values = [0, 1]  # Assuming 0: Low, 1: High education
    informed_values = [0, 1]  # 0: Not informed, 1: Informed
    health_values = [1, 2]  # Health states 1 and 2 only

    # Labels for plotting
    sex_labels = ["Male", "Female"]
    education_labels = ["Low Edu", "High Edu"]
    informed_labels = ["Not Informed", "Informed"]
    health_labels = ["Health 1", "Health 2"]

    # Create large subplot grid: 2 rows (health) × 4 cols (sex x education)
    fig, axes = plt.subplots(2, 4, figsize=get_figsize(4, 2))
    fig.suptitle(
        "Retirement Experience Law of Motion - All Demographics",
        fontsize=16,
        fontweight="bold",
    )

    for health_idx, health in enumerate(health_values):
        col = 0
        for sex in sex_values:
            for education in education_values:

                ax = axes[health_idx, col]

                # Plot for both informed states
                for informed_idx, informed in enumerate(informed_values):

                    # Plot different experience levels
                    for i, exp_years in enumerate(np.arange(30, 50, 10)):
                        # Scale the experience for a not retired person last period
                        exp = scale_experience_years(
                            experience_years=exp_years,
                            period=periods - 1,
                            is_retired=np.zeros_like(periods, dtype=bool),
                            model_specs=specs,
                        )

                        exp_next = get_next_period_experience(
                            period=periods,
                            lagged_choice=0,
                            policy_state=8,  # Policy state 8 as in original
                            sex=sex,
                            education=education,
                            experience=exp,
                            informed=informed,
                            health=health,
                            model_specs=specs,
                        )

                        exp_years_next = construct_experience_years(
                            float_experience=exp_next,
                            period=periods,
                            is_retired=np.ones_like(periods, dtype=bool),
                            model_specs=specs,
                        )

                        # Use different line styles for informed status
                        linestyle = "-" if informed == 1 else "--"
                        alpha = 1.0 if informed == 1 else 0.7

                        ax.plot(
                            periods + 30,
                            exp_years_next,
                            color=colors[i % len(colors)],
                            linestyle=linestyle,
                            alpha=alpha,
                            linewidth=1.5,
                        )

                # Formatting for each subplot
                subplot_title = (
                    f"{sex_labels[sex]}, {education_labels[education]}\n"
                    f"{health_labels[health - 1]}"
                )
                ax.set_title(subplot_title, fontsize=10)
                ax.grid(True, alpha=0.2)

                # Only add axis labels to edge subplots
                if health_idx == 1:  # Bottom row
                    ax.set_xlabel("Period", fontsize=10)
                if col == 0:  # Left column
                    ax.set_ylabel("Experience Years", fontsize=10)

                col += 1

    # Create custom legend with both experience levels and informed status
    exp_years_values = np.arange(30, 50, 10)
    legend_elements = []

    # Add experience level legends
    for i, exp_years in enumerate(exp_years_values):
        legend_elements.append(
            plt.Line2D(
                [0], [0], color=colors[i % len(colors)], label=f"Exp {exp_years}"
            )
        )

    # Add separator and informed status legends
    legend_elements.append(plt.Line2D([0], [0], color="white", label=""))  # Spacer
    legend_elements.append(
        plt.Line2D([0], [0], color="black", linestyle="-", label="Informed")
    )
    legend_elements.append(
        plt.Line2D(
            [0], [0], color="black", linestyle="--", alpha=0.7, label="Not Informed"
        )
    )

    fig.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(0.98, 0.5))

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legend

    if save:
        filename_base = "retirement_experience_law_of_motion_all_demographics"
        fig.savefig(
            path_dict["model_plots"] + f"{filename_base}.pdf", bbox_inches="tight"
        )
        fig.savefig(
            path_dict["model_plots"] + f"{filename_base}.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
