import matplotlib.pyplot as plt
import numpy as np
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_job_transitions(path_dict, params):
    """Plot job separation probabilities."""
    specs = generate_derived_and_data_derived_specs(path_dict)

    n_working_periods = specs["max_ret_age"] - specs["start_age"] + 1
    n_education_types = specs["n_education_types"]
    n_sexes = specs["n_sexes"]
    job_sep_probs = np.zeros(
        (n_sexes, n_education_types, n_working_periods), dtype=float
    )
    job_offer_probs = np.zeros(
        (n_sexes, n_education_types, n_working_periods), dtype=float
    )

    for sex in range(n_sexes):
        for edu in range(n_education_types):
            for period in range(n_working_periods):
                job_sep_probs[sex, edu, period] = job_offer_process_transition(
                    params=params,
                    options=specs,
                    sex=sex,
                    education=edu,
                    period=period,
                    choice=2,
                )[0]
                job_offer_probs[sex, edu, period] = job_offer_process_transition(
                    params=params,
                    options=specs,
                    sex=sex,
                    education=edu,
                    period=period,
                    choice=1,
                )[1]

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax1.plot(
                np.arange(n_working_periods),
                job_sep_probs[sex_var, edu_var, :],
                label=f"{sex_label}; {edu_label}",
            )
            ax2.plot(
                np.arange(n_working_periods),
                job_offer_probs[sex_var, edu_var, :],
                label=f"{sex_label}; {edu_label}",
            )

    ax1.set_title("Job destruction rates")
    ax2.set_title("Job offer rates at start params (job finding rate)")
    ax1.set_xlabel("Period")
    ax2.set_xlabel("Period")

    ax1.legend()
    ax2.legend()
