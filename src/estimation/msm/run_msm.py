# Set paths of project
import pickle as pkl

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
from estimation.msm.msm_estimation_setup import estimate_model
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

params_to_estimate_names = [
    # "mu_men",
    # Men Full-time - 4 parameters
    "disutil_ft_work_good_men",
    "disutil_ft_work_bad_men",
    # Men unemployment - 2 parameters
    "disutil_unemployed_good_men",
    "disutil_unemployed_bad_men",
    # Taste shock men - 1 parameter
    # "taste_shock_scale_men",
    # # Men job finding - 3 parameters
    # "job_finding_logit_const_men",
    # "job_finding_logit_high_educ_men",
    # "job_finding_logit_good_health_men",
    # "job_finding_logit_above_50_men",
    # "job_finding_logit_above_55_men",
    # "job_finding_logit_above_60_men",
]
model_name = "msm_first"
LOAD_LAST_ESTIMATE = False
LOAD_SOL_MODEL = True
SAVE_RESULTS = True
USE_WEIGHTS = False

if LOAD_LAST_ESTIMATE:
    last_estimate = pkl.load(
        open(paths_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )
else:
    last_estimate = None


# Load start params
start_params_all = load_and_set_start_params(paths_dict)

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    # file_append=model_name,
    start_params_all=start_params_all,
    load_model=LOAD_SOL_MODEL,
    # use_weights=USE_WEIGHTS,
    last_estimate=last_estimate,
    # save_results=SAVE_RESULTS,
)
print(estimation_results)


# %%
