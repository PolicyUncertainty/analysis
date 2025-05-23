import matplotlib.pyplot as plt
import numpy as np

from export_results.figures.color_map import JET_COLOR_MAP
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from process_data.first_step_sample_scripts.create_job_sep_sample import (
    create_job_sep_sample,
)
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_job_transitions(path_dict, specs):
    """Plot job separation probabilities."""

    n_working_periods = 65 - specs["start_age"] + 1

    df_job = create_job_sep_sample(path_dict, specs, load_data=True)

    obs_shares = df_job.groupby(["sex", "education", "age"])["job_sep"].mean()
    working_ages = np.arange(n_working_periods) + specs["start_age"]

    df_job["good_health"] = (
        df_job["lagged_health"] == specs["good_health_var"]
    ).astype(int)
    df_job["predicted_probs"] = specs["job_sep_probs"][
        df_job["sex"].values,
        df_job["education"].values,
        df_job["good_health"].values,
        df_job["age"].values,
    ]

    # n_education_types = specs["n_education_types"]
    # n_sexes = specs["n_sexes"]
    # job_offer_probs = np.zeros(
    #     (n_sexes, n_education_types, n_working_periods), dtype=float
    # )

    # for sex in range(n_sexes):
    #     for edu in range(n_education_types):
    #         for period in range(n_working_periods):
    # job_offer_probs[sex, edu, period] = job_offer_process_transition(
    #     params=params,
    #     options=specs,
    #     sex=sex,
    #     education=edu,
    #     period=period,
    #     choice=1,
    # )[1]

    fig, axs = plt.subplots(ncols=2, figsize=(12, 8))
    predited_probs = df_job.groupby(["sex", "education", "age"])[
        "predicted_probs"
    ].mean()
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axs[sex_var]
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax.plot(
                working_ages,
                predited_probs.loc[(sex_var, edu_var, working_ages)],
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
