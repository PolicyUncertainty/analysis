import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_results.figures.color_map import JET_COLOR_MAP
from process_data.first_step_sample_scripts.create_job_sep_sample import (
    create_job_sep_sample,
)


def plot_job_separations(path_dict, specs):
    """Plot job separation probabilities."""

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
                color=JET_COLOR_MAP[edu_var],
            )
            ax.plot(
                working_ages,
                obs_shares.loc[(sex_var, edu_var, working_ages)],
                label=f"Obs. {edu_label}",
                linestyle="--",
                color=JET_COLOR_MAP[edu_var],
            )

        ax.set_title(f"{sex_label}")
        ax.set_xlabel("Age")
        ax.set_ylim([0, 0.1])

    axs[0].legend(loc="upper left")
    fig.savefig(path_dict["plots"] + "job_separation.png")


def plot_job_offer_transitions(path_dict, specs, params):
    """Plot job offer transition probabilities."""

    n_working_periods = 65 - specs["start_age"] + 1
    working_ages = np.arange(n_working_periods) + specs["start_age"]

    df = pd.read_csv(path_dict["struct_est_sample"])

    df["age"] = df["period"] + specs["start_age"]
    df_ue = df[df["lagged_choice"] == 1].copy()
    df_ue["work_start"] = df_ue["choice"].isin([2, 3]).astype(int)

    observed_probs = df_ue.groupby(["sex", "age"])["work_start"].mean()

    age = df_ue["age"].values
    education = df_ue["education"].values
    good_health = (df_ue["health"] == specs["good_health_var"]).astype(int).values

    exp_value = np.exp(
        params["job_finding_logit_const_men"]
        # + params["job_finding_logit_age_men"] * age
        + params["job_finding_logit_high_educ_men"] * education
        + params["job_finding_logit_good_health_men"] * good_health
        + params["job_finding_logit_above_50_men"] * (age >= 50)
        + params["job_finding_logit_above_55_men"] * (age >= 55)
        + params["job_finding_logit_above_60_men"] * (age >= 60)
    )
    df_ue["job_offer_probs"] = exp_value / (1 + exp_value)

    predicted_probs = df_ue.groupby(["sex", "age"])["job_offer_probs"].mean()

    fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axs[sex_var]
        # for edu_var, edu_label in enumerate(specs["education_labels"]):
        ax.plot(
            working_ages,
            predicted_probs.loc[(sex_var, working_ages)],
            label=f"Est.",
            color=JET_COLOR_MAP[0],
        )
        ax.plot(
            working_ages,
            observed_probs.loc[(sex_var, working_ages)],
            label=f"Obs.",
            linestyle="--",
            color=JET_COLOR_MAP[0],
        )

        ax.set_title(f"{sex_label}")
        ax.set_xlabel("Age")
        ax.set_ylim([0, 0.8])

        axs[0].legend(loc="upper left")
    fig.savefig(path_dict["plots"] + "job_offers.png")

    # fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
    # for sex_var, sex_label in enumerate(specs["sex_labels"]):
    #     ax = axs[sex_var]
    #     for edu_var, edu_label in enumerate(specs["education_labels"]):
    #         ax.plot(
    #             working_ages,
    #             predicted_probs.loc[(sex_var, edu_var, working_ages)],
    #             label=f"Est. {edu_label}",
    #             color=JET_COLOR_MAP[edu_var],
    #         )
    #         ax.plot(
    #             working_ages,
    #             observed_probs.loc[(sex_var, edu_var, working_ages)],
    #             label=f"Obs. {edu_label}",
    #             linestyle="--",
    #             color=JET_COLOR_MAP[edu_var],
    #         )
    #
    #     ax.set_title(f"{sex_label}")
    #     ax.set_xlabel("Age")
    #     ax.set_ylim([0, 0.8])
    #
    # axs[0].legend(loc="upper left")
    # fig.savefig(path_dict["plots"] + "job_offers.png")
