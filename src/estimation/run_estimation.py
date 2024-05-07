# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")

from set_paths import create_path_dict

do_first_step = input("Do first step estimation? (y/n): ") == "y"
if do_first_step:
    estimate_wage_only = input("Estimate wage only? (y/n): ") == "y"
do_model_estimatation = input("Estimate model? (y/n): ") == "y"
paths_dict = create_path_dict(analysis_path, define_user=do_first_step)

# Import jax and set jax to work with 64bit
import jax

jax.config.update("jax_enable_x64", True)
import pandas as pd
from estimation.estimate_setup import estimate_model


if do_first_step:
    # Estimate parameters of SRA truncated normal distributions
    from estimation.first_step_estimation.est_SRA_expectations import (
        estimate_truncated_normal,
    )
    from model_code.derive_specs import read_and_derive_specs

    specs = read_and_derive_specs(paths_dict["specs"])

    if not estimate_wage_only:
        df_exp_policy_dist = estimate_truncated_normal(
            paths_dict, specs, load_data=False
        )

        # Estimate SRA random walk
        from estimation.first_step_estimation.est_SRA_random_walk import (
            estimate_expected_SRA_variance,
            est_expected_SRA,
        )

        est_expected_SRA(paths_dict)
        estimate_expected_SRA_variance(paths_dict)

    # Estimate wage parameter
    from estimation.first_step_estimation.est_wage_equation import (
        estimate_wage_parameters,
    )

    estimate_wage_parameters(paths_dict)
    # from estimation.first_step_estimation.est_wage_equation import plot_wages_by_edu
    # plot_wages_by_edu(wage_parameters)

if do_model_estimatation:
    estimation_results = estimate_model(paths_dict, load_model=True)
    print(estimation_results)

# %%
