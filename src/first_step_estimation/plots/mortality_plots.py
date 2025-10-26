# Description: This file contains plotting functions for mortality estimation results.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import get_figsize, set_colors, set_plot_defaults


def plot_mortality(path_dict, specs, show=False, save=False, paper_plot=False):
    """Plot mortality characteristics.

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
    paper_plot : bool, default False
        Whether to create separate figures for paper
    """
    set_plot_defaults()
    # Load the data
    # Mortality estimation sample
    df = pd.read_pickle(
        path_dict["first_step_data"] + "mortality_transition_estimation_sample.pkl"
    )

    # Estimated mortality transition matrix (life table adjusted probabilities of death)
    estimated_mortality = pd.read_csv(
        path_dict["first_step_results"] + "mortality_transition_matrix.csv"
    )

    # Estimated mortality parameters
    df_params_male = pd.read_csv(
        path_dict["first_step_results"] + "est_params_mortality_men.csv"
    )
    df_params_male.set_index("Unnamed: 0", inplace=True)
    df_params_female = pd.read_csv(
        path_dict["first_step_results"] + "est_params_mortality_women.csv"
    )
    df_params_female.set_index("Unnamed: 0", inplace=True)

    observed_health_vars = specs["observed_health_vars"]

    # Generate out cols in estimated mortality
    estimated_mortality["survival_prob_year"] = np.nan
    estimated_mortality["survival_prob"] = np.nan

    colors, _ = set_colors()

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for health in observed_health_vars:
            for edu_var, edu_label in enumerate(specs["education_labels"]):
                # Health label of alive state
                mask = (
                    (estimated_mortality["sex"] == sex_var)
                    & (estimated_mortality["health"] == health)
                    & (estimated_mortality["education"] == edu_var)
                )

                # Filter the data for the current combination
                filtered_data = estimated_mortality.loc[mask,]

                estimated_mortality.loc[mask, "survival_prob_year"] = (
                    1 - filtered_data["death_prob"]
                )

                estimated_mortality.loc[mask, "survival_prob"] = np.cumprod(
                    estimated_mortality.loc[mask, "survival_prob_year"]
                )

    # Create plots showing survival probabilities
    if paper_plot:
        figs = []
        axs = []
        for _ in range(4):
            fig, ax = plt.subplots()
            figs.append(fig)
            axs.append(ax)
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=get_figsize(2, 2))

    titles = []
    i = 0
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for health, health_label in enumerate(
            specs["health_labels"][:2]
        ):  # Only observed health states
            if paper_plot:
                ax = axs[i]
                sex_label_lower = sex_label.lower()
                health_label_lower = health_label.lower().replace(" ", "_")
                titles.append(f"mortality_{sex_label_lower}_{health_label_lower}")
                i += 1
            else:
                ax = axs[sex_var, health]

            for edu_var, edu_label in enumerate(specs["education_labels"]):
                mask = (
                    (estimated_mortality["sex"] == sex_var)
                    & (estimated_mortality["health"] == health)
                    & (estimated_mortality["education"] == edu_var)
                )

                filtered_data = estimated_mortality.loc[mask]

                if not filtered_data.empty:
                    ax.plot(
                        filtered_data["age"],
                        filtered_data["survival_prob"],
                        color=colors[edu_var],
                        label=f"{edu_label}",
                        linewidth=2,
                    )

            ax.set_xlabel("Age")
            ax.set_ylabel("Survival Probability")
            ax.set_ylim(0, 1)
            ax.legend(frameon=False)

            if not paper_plot:
                ax.set_title(f"{sex_label}, {health_label}")

    if paper_plot:
        for fig, title in zip(figs, titles):
            fig.tight_layout()
            fig.savefig(
                path_dict["first_step_plots"] + f"{title}.png",
                bbox_inches="tight",
                dpi=300,
            )
    else:
        fig.tight_layout()
        if save:
            fig.savefig(
                path_dict["first_step_plots"] + "mortality_survival.pdf",
                bbox_inches="tight",
            )
            fig.savefig(
                path_dict["first_step_plots"] + "mortality_survival.png",
                bbox_inches="tight",
                dpi=300,
            )

    if show:
        plt.show()
    else:
        if paper_plot:
            for fig in figs:
                plt.close(fig)
        else:
            plt.close(fig)


def plot_mortality_hazard_ratios(path_dict, specs, show=False, save=False):
    """Plot mortality hazard ratios by demographic groups.

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

    # Load estimated mortality parameters
    param_files = ["est_params_mortality_men.csv", "est_params_mortality_women.csv"]
    sex_labels = specs["sex_labels"]

    fig, axs = plt.subplots(ncols=2, figsize=get_figsize(2, 1))

    for sex_var, (param_file, sex_label) in enumerate(zip(param_files, sex_labels)):
        try:
            df_params = pd.read_csv(
                path_dict["first_step_results"] + param_file, index_col=0
            )

            ax = axs[sex_var]

            # Plot hazard ratios for each parameter
            for i, (param_name, row) in enumerate(df_params.iterrows()):
                if param_name != "age":  # Skip age parameter
                    hazard_ratio = (
                        row["hazard_ratio"]
                        if "hazard_ratio" in df_params.columns
                        else np.exp(row["value"])
                    )

                    ax.bar(
                        i,
                        hazard_ratio,
                        color=colors[i % len(colors)],
                        label=param_name,
                        alpha=0.7,
                    )

            ax.axhline(
                y=1, color="black", linestyle="--", alpha=0.5, label="No effect (HR=1)"
            )
            ax.set_title(f"Mortality Hazard Ratios - {sex_label}")
            ax.set_ylabel("Hazard Ratio")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        except FileNotFoundError:
            ax = axs[sex_var]
            ax.text(
                0.5,
                0.5,
                f"Parameter file not found:\n{param_file}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Mortality Hazard Ratios - {sex_label}")

    fig.tight_layout()

    if save:
        fig.savefig(
            path_dict["first_step_plots"] + "mortality_hazard_ratios.pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            path_dict["first_step_plots"] + "mortality_hazard_ratios.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
