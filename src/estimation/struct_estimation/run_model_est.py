# Set paths of project
import pickle as pkl

import pandas as pd
from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)

from estimation.struct_estimation.estimate_setup import estimate_model

params_to_estimate_names = [
    # "mu",
    "disutil_unemployed_men",
    "disutil_ft_work_good_men",
    "disutil_ft_work_bad_men",
    "disutil_pt_work_good_women",
    "disutil_ft_work_good_women",
    "disutil_ft_work_bad_women",
    "disutil_pt_work_bad_women",
    "disutil_unemployed_women",
    # "disutil_children_pt_work",
    "disutil_children_ft_work_low",
    "disutil_children_ft_work_high",
    # "disutil_not_retired_bad",
    # "disutil_working_bad",
    # "disutil_not_retired_good",
    # "disutil_working_good",
    # "bequest_scale",
    "taste_shock_scale_men",
    "taste_shock_scale_women",
    "job_finding_logit_const_men",
    "job_finding_logit_age_men",
    "job_finding_logit_high_educ_men",
    "job_finding_logit_const_women",
    "job_finding_logit_age_women",
    "job_finding_logit_high_educ_women",
]

last_estimate = pkl.load(open(paths_dict["est_results"] + "est_params_pete.pkl", "rb"))
last_estimate["taste_shock_scale_men"] = 0.3
last_estimate["taste_shock_scale_women"] = last_estimate["lambda"]
del last_estimate["lambda"]
del last_estimate["disutil_unemployed_men"]
del last_estimate["disutil_ft_work_good_men"]
del last_estimate["disutil_ft_work_bad_men"]

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append="taste",
    slope_disutil_method=False,
    load_model=False,
    last_estimate=last_estimate,
    save_results=False,
)
print(estimation_results)


# %%
