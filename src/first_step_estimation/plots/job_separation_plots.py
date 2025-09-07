# Description: This file contains plotting functions for job separation estimation results.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from set_styles import set_colors
from process_data.first_step_sample_scripts.create_job_sep_sample import (
    create_job_sep_sample,
)


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
    n_working_periods = 65 - specs["start_age"] + 1
    working_ages = np.arange(n_working_periods) + specs["start_age"]

    df_job = create_job_sep_sample(path_dict, specs, load_data=True)

    obs_shares = df_job.groupby(["sex", "education", "age"])["job_sep"].mean()

    df_job["good_health"] = (
        df_job["lagged_health"] == specs["good_health_var"]
    ).astype(int)
    df_job["predicted_probs"] = specs["job_sep_probs"][
        df_job["sex"].values,
        df_job["education"].values,
        df_job["good_health"].values,
        df_job["age"].values,
    ]

    fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
    colors, _ = set_colors()
    
    predicted_probs = df_job.groupby(["sex", "education", "age"])[
        "predicted_probs"
    ].mean()
    
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axs[sex_var]
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax.plot(
                working_ages,
                predicted_probs.loc[(sex_var, edu_var, working_ages)],
                label=f"Est. {edu_label}",
                color=colors[edu_var],
            )
            ax.plot(
                working_ages,
                obs_shares.loc[(sex_var, edu_var, working_ages)],
                label=f"Obs. {edu_label}",
                linestyle="--",
                color=colors[edu_var],
            )
            
        ax.set_title(f"{sex_label}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Job Separation Probability")
        ax.legend()
        
    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["first_step_plots"] + "job_separations.pdf", bbox_inches="tight")
        fig.savefig(path_dict["first_step_plots"] + "job_separations.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)