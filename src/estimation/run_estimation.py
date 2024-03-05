# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")

from set_paths import create_path_dict

do_first_step = input("Do first step estimation? (y/n): ") == "y"
do_model_estimatation = input("Estimate model? (y/n): ") == "y"
paths_dict = create_path_dict(analysis_path, define_user=do_first_step)

# Import jax and set jax to work with 64bit
import jax

jax.config.update("jax_enable_x64", True)
import pandas as pd
from estimation.estimate_setup import estimate_model


if do_first_step:
    # Estimate policy expectation
    from estimation.first_step_estimation.est_SRA_random_walk import (
        estimate_expected_SRA_variance,
        est_expected_SRA,
    )

    data_expectation = pd.read_pickle(
        paths_dict["intermediate_data"] + "policy_expect_data.pkl"
    )
    alpha_hat = est_expected_SRA(paths_dict, data_expectation)
    sigma_sq_hat = estimate_expected_SRA_variance(paths_dict, data_expectation)

    policy_params = {"alpha_hat": alpha_hat, "sigma_sq_hat": sigma_sq_hat}

    # Estimate wage parameter
    from estimation.first_step_estimation.est_wage_equation import (
        estimate_wage_parameters,
    )

    wage_parameters = estimate_wage_parameters(paths_dict)

if do_model_estimatation:
    estimation_results = estimate_model(paths_dict, load_model=True)


