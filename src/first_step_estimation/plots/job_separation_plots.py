# Description: This file contains plotting functions for job separation estimation results.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_data.first_step_sample_scripts.create_job_sep_sample import (
    create_job_sep_sample,
)
from set_styles import set_colors, get_figsize


def plot_job_separations(path_dict, specs, show=False, save=False):
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
    """
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

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=get_figsize(2, 2))
    colors, _ = set_colors()

    obs_shares = df_job.groupby(["sex", "education", "good_health", "age"])[
        "job_sep"
    ].mean()
    predicted_probs = df_job.groupby(["sex", "education", "good_health", "age"])[
        "predicted_probs"
    ].mean()
    health_labels = ["Bad Health", "Good Health"]
    for edu_var, edu_label in enumerate(specs["education_labels"]):
        for sex_var, sex_label in enumerate(specs["sex_labels"]):
            ax = axs[sex_var, edu_var]
            for good_health in [0, 1]:
                ax.plot(
                    working_ages,
                    predicted_probs.loc[(sex_var, edu_var, good_health, working_ages)],
                    label=f"Est. {health_labels[good_health]}",
                    color=colors[good_health],
                )
                ax.plot(
                    working_ages,
                    obs_shares.loc[(sex_var, edu_var, good_health, working_ages)],
                    # label=f"Obs. {edu_label}; {health_labels[good_health]}",
                    linestyle="--",
                    color=colors[good_health],
                )

        axs[0, edu_var].set_title(f"{edu_label}")
    axs[1, 0].set_xlabel("Age")
    axs[1, 1].set_xlabel("Age")

    axs[0, 0].set_ylabel("Job Separation Probability")
    axs[1, 0].set_ylabel("Job Separation Probability")
    axs[0, 0].legend()
    axs[0, 1].legend()

    fig.tight_layout()

    if save:
        fig.savefig(
            path_dict["first_step_plots"] + "job_separations.pdf", bbox_inches="tight"
        )
        fig.savefig(
            path_dict["first_step_plots"] + "job_separations.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
