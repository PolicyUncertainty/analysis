import numpy as np
import matplotlib.pyplot as plt
from model_code.derive_specs import generate_derived_and_data_derived_specs
from model_code.job_offers import job_offer_process_transition
import yaml
from estimation.estimate_setup import create_job_offer_params_from_start


def plot_job_separation(path_dict):
    """Plot job separation probabilities."""
    specs = generate_derived_and_data_derived_specs(path_dict)

    start_params_all = yaml.safe_load(open(path_dict["start_params"], "rb"))

    job_sep_params = create_job_offer_params_from_start(path_dict)
    start_params_all.update(job_sep_params)


    n_working_periods = 45
    n_education_types = specs["n_education_types"]
    job_sep_probs = np.zeros((n_education_types, n_working_periods))
    job_offer_probs = np.zeros((n_education_types, n_working_periods))

    for edu in range(n_education_types):
        for period in range(n_working_periods):
            job_sep_probs[edu, period] = job_offer_process_transition(
                params=start_params_all,
                options=specs,
                education=edu,
                period=period,
                choice=1
            )[0]
            job_offer_probs[edu, period] = job_offer_process_transition(
                params=start_params_all,
                options=specs,
                education=edu,
                period=period,
                choice=0
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