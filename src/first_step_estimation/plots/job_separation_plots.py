# Description: This file contains plotting functions for job separation estimation results.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_data.first_step_sample_scripts.create_job_sep_sample import (
    create_job_sep_sample,
)
from set_styles import get_figsize, set_colors, set_plot_defaults


def plot_job_separations(path_dict, specs, show=False, save=False, paper_plot=False):
    """Plot job separation probabilities.

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
    n_working_periods = 65 - specs["start_age"]
    working_ages = np.arange(n_working_periods) + specs["start_age"]

    df_job = create_job_sep_sample(path_dict, specs, load_data=True)

    df_job["good_health"] = (
        df_job["lagged_health"] == specs["good_health_var"]
    ).astype(int)
    df_job["predicted_probs"] = specs["job_sep_probs"][
        df_job["sex"].values,
        df_job["education"].values,
        df_job["good_health"].values,
        df_job["age"].values,
    ]

    if paper_plot:
        figs = []
        axs = []
        for _ in range(2):
            fig, ax = plt.subplots()
            figs.append(fig)
            axs.append(ax)
    else:
        fig, axs = plt.subplots(ncols=2, figsize=get_figsize(ncols=2))

    colors, _ = set_colors()

    obs_shares = df_job.groupby(["sex", "education", "age"])["job_sep"].mean()
    predicted_probs = df_job.groupby(["sex", "education", "age"])[
        "predicted_probs"
    ].mean()

    # Create state labels combining education and health
    state_labels = []
    for edu_label in specs["education_labels"]:
        for health_label in ["bad health", "good health"]:
            state_labels.append(f"{edu_label.lower()}, {health_label}")

    titles = []
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        if paper_plot:
            ax = axs[sex_var]
            sex_label_lower = sex_label.lower()
            titles.append(f"job_sep_{sex_label_lower}")
        else:
            ax = axs[sex_var]

        color_idx = 0
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax.plot(
                working_ages,
                predicted_probs.loc[(sex_var, edu_var, working_ages)],
                label=f"est. {state_labels[color_idx]}",
                color=colors[color_idx],
            )
            ax.plot(
                working_ages,
                obs_shares.loc[(sex_var, edu_var, working_ages)],
                label=f"obs. {state_labels[color_idx]}",
                linestyle="--",
                color=colors[color_idx],
            )
            color_idx += 1

        ax.set_ylim([0, 0.08])
        ax.legend(frameon=False)
        ax.set_xlabel("Age")
        ax.set_ylabel("Job Separation Probability")

        if not paper_plot:
            ax.set_title(f"{sex_label}")

    if paper_plot:
        for fig, title in zip(figs, titles):
            fig.tight_layout()
            fig.savefig(
                path_dict["first_step_plots"] + f"{title}.png",
                bbox_inches="tight",
                dpi=100,
            )
    else:
        fig.tight_layout()
        if save:
            fig.savefig(
                path_dict["first_step_plots"] + "job_separations.pdf",
                bbox_inches="tight",
            )
            fig.savefig(
                path_dict["first_step_plots"] + "job_separations.png",
                bbox_inches="tight",
                dpi=100,
            )

    if show:
        plt.show()
    else:
        if paper_plot:
            for fig in figs:
                plt.close(fig)
        else:
            plt.close(fig)
