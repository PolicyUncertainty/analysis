import matplotlib.pyplot as plt
import numpy as np
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_job_separation(path_dict, params):
    """Plot job separation probabilities."""
    specs = generate_derived_and_data_derived_specs(path_dict)

    n_working_periods = 45
    n_education_types = specs["n_education_types"]
    job_sep_probs = np.zeros((n_education_types, n_working_periods))
    job_offer_probs = np.zeros((n_education_types, n_working_periods))

    for edu in range(n_education_types):
        for period in range(n_working_periods):
            job_sep_probs[edu, period] = job_offer_process_transition(
                params=params,
                options=specs,
                education=edu,
                period=period,
                choice=2,
            )[0]
            job_offer_probs[edu, period] = job_offer_process_transition(
                params=params,
                options=specs,
                education=edu,
                period=period,
                choice=1,
            )[1]

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    for edu in range(n_education_types):
        ax1.plot(
            np.arange(n_working_periods),
            job_sep_probs[edu, :],
            label=f"Education {edu}",
        )
        ax2.plot(
            np.arange(n_working_periods),
            job_offer_probs[edu, :],
            label=f"Education {edu}",
        )

    ax1.set_title("Job destruction rates")
    ax2.set_title("Job offer rates at start params (job finding rate)")

    ax1.legend()
    ax2.legend()
