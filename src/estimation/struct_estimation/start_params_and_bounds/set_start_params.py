import pandas as pd
import yaml

from estimation.struct_estimation.start_params_and_bounds.disability_state_start_params import (
    est_disability_prob,
)
from estimation.struct_estimation.start_params_and_bounds.job_offer_start_params import (
    est_job_offer_params_full_obs,
)
from specs.derive_specs import generate_derived_and_data_derived_specs


def load_and_set_start_params(path_dict):
    start_params_all = yaml.safe_load(
        open(path_dict["start_params_and_bounds"] + "start_params.yaml", "rb")
    )

    # Create start values for job offer probabilities
    struct_est_sample = pd.read_csv(path_dict["struct_est_sample"])
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=False)

    # Estimate start values for job offers
    job_offer_params = est_job_offer_params_full_obs(struct_est_sample, specs)
    # Update start values
    start_params_all.update(job_offer_params)

    # Estimate disability probability start params
    disability_prob_params = est_disability_prob(path_dict, specs)
    # Update start values
    start_params_all.update(disability_prob_params)

    return start_params_all
