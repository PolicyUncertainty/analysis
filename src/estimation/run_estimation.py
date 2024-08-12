# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")

from model_code.derive_specs import read_and_derive_specs
from set_paths import create_path_dict

# do_first_step = input("Do first step estimation? (y/n): ") == "y"
estimate_sra = input("Estimate SRA process? (y/n): ") == "y"
estimate_wage = input("Estimate wage? (y/n): ") == "y"
estimate_partner_wage = input("Estimate partner wage? (y/n): ") == "y"
estimate_job_sep = input("Estimate job separation? (y/n): ") == "y"
do_model_estimatation = input("Estimate model? (y/n): ") == "y"


paths_dict = create_path_dict(analysis_path, define_user=estimate_sra)
specs = read_and_derive_specs(paths_dict["specs"])


if estimate_sra:
    # Estimate parameters of SRA truncated normal distributions
    from estimation.first_step_estimation.est_SRA_expectations import (
        estimate_truncated_normal,
    )

    df_exp_policy_dist = estimate_truncated_normal(paths_dict, specs, load_data=False)

    # Estimate SRA random walk
    from estimation.first_step_estimation.est_SRA_random_walk import (
        est_SRA_params,
    )

    est_SRA_params(paths_dict)

if estimate_wage:
    # Estimate wage parameters
    # Average wage parameters are estimated to compute education-specific pensions
    from estimation.first_step_estimation.est_wage_equation import (
        estimate_wage_parameters,
        estimate_average_wage_parameters,
    )

    estimate_wage_parameters(paths_dict)
    estimate_average_wage_parameters(paths_dict)

if estimate_partner_wage:
    # Estimate partner wage parameters
    from estimation.first_step_estimation.est_partner_wage_equation import (
        estimate_partner_wage_parameters,
    )

    estimate_partner_wage_parameters(paths_dict)

if estimate_job_sep:
    # Estimate job separation
    from estimation.first_step_estimation.est_job_sep import est_job_sep

    est_job_sep(paths_dict, specs, load_data=True)

if do_model_estimatation:
    from estimation.estimate_setup import estimate_model

    estimation_results = estimate_model(paths_dict, load_model=False)
    print(estimation_results)

# %%
