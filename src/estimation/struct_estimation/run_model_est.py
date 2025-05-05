# Set paths of project
import pickle as pkl

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)

from estimation.struct_estimation.map_params_to_current import gender_separate_models
from estimation.struct_estimation.scripts.estimate_setup import estimate_model
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

params_to_estimate_names = [
    # "mu_men",
    # Men Full-time - 4 parameters
    "disutil_ft_work_high_good_men",
    "disutil_ft_work_high_bad_men",
    "disutil_ft_work_low_good_men",
    "disutil_ft_work_low_bad_men",
    # Men unemployment - 2 parameters
    "disutil_unemployed_high_men",
    "disutil_unemployed_low_men",
    # Taste shock men - 1 parameter
    # "taste_shock_scale_men",
    # Men job finding - 3 parameters
    # "job_finding_logit_const_men",
    # "job_finding_logit_age_men",
    # "job_finding_logit_high_educ_men",
    # Disability probability men - 3 parameters
    # "disability_logit_const_men",
    # "disability_logit_age_men",
    # "disability_logit_high_educ_men",
    # "mu_women",
    # # # Women Full-time - 4 parameters
    # "disutil_ft_work_high_good_women",
    # "disutil_ft_work_high_bad_women",
    # "disutil_ft_work_low_good_women",
    # "disutil_ft_work_low_bad_women",
    # # # Women Part-time - 4 parameters
    # "disutil_pt_work_high_good_women",
    # "disutil_pt_work_high_bad_women",
    # "disutil_pt_work_low_good_women",
    # "disutil_pt_work_low_bad_women",
    # # Women Unemployment - 2 parameters
    # "disutil_unemployed_high_women",
    # "disutil_unemployed_low_women",
    # # Children - 2 parameters
    # "disutil_children_ft_work_low",
    # "disutil_children_ft_work_high",
    # # Taste shock women - 1 parameter
    # "taste_shock_scale_women",
    # # Women job finding - 3 parameters
    # "job_finding_logit_const_women",
    # "job_finding_logit_age_women",
    # "job_finding_logit_high_educ_women",
    # # Disability probability women - 3 parameters
    # "disability_logit_const_women",
    # "disability_logit_age_women",
    # "disability_logit_high_educ_women",
]
model_name = "disability"
LOAD_LAST_ESTIMATE = True
LOAD_SOL_MODEL = True
SAVE_RESULTS = True
USE_WEIGHTS = False

if LOAD_LAST_ESTIMATE:
    last_estimate = pkl.load(
        open(paths_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )
    last_estimate = gender_separate_models(last_estimate)
    start_params_all = load_and_set_start_params(paths_dict)
    for key in last_estimate.keys():
        if "men" in key:
            last_estimate[key] = start_params_all[key]

else:
    last_estimate = None

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append=model_name,
    load_model=LOAD_SOL_MODEL,
    use_weights=USE_WEIGHTS,
    last_estimate=last_estimate,
    save_results=SAVE_RESULTS,
)
print(estimation_results)


# %%
