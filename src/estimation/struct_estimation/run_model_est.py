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
    "disutil_unemployed_men",
    "disutil_pt_work_good_women",
    "disutil_ft_work_good_women",
    "disutil_ft_work_bad_women",
    "disutil_pt_work_bad_women",
    "disutil_unemployed_women",
    # "disutil_children_pt_work",
    "disutil_children_ft_work",
    # "disutil_not_retired_bad",
    # "disutil_working_bad",
    # "disutil_not_retired_good",
    # "disutil_working_good",
    # "bequest_scale",
    "lambda",
    "job_finding_logit_const_men",
    "job_finding_logit_age_men",
    "job_finding_logit_high_educ_men",
    "job_finding_logit_const_women",
    "job_finding_logit_age_women",
    "job_finding_logit_high_educ_women",
]


last_estimate = {
    "lambda": 0.8774365900160864,
    "disutil_ft_work_good_men": 0.4874478986458445,
    "disutil_ft_work_bad_men": 1.4844829806153268,
    "disutil_unemployed_men": 0.2,
    "job_finding_logit_const_men": 0.6969263997261611,
    "job_finding_logit_age_men": -0.06920021523595966,
    "job_finding_logit_high_educ_men": 0.14929353536195986,
    "disutil_ft_work_good_women": 2.0981866690301305,
    "disutil_ft_work_bad_women": 2.3611125791635446,
    "disutil_children_ft_work": 0.4133940938632231,
    "disutil_children_pt_work": 1e-12,
    "disutil_unemployed_women": 1.5,
    "disutil_pt_work_good_women": 2.2060519982921734,
    "disutil_pt_work_bad_women": 2.4979321773682117,
    "job_finding_logit_const_women": 0.30698425755100517,
    "job_finding_logit_age_women": -0.13337968578724865,
    "job_finding_logit_high_educ_women": 2.3886587852919723,
}

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append="pete",
    slope_disutil_method=False,
    load_model=True,
    last_estimate=last_estimate,
    save_results=False,
)
print(estimation_results)


# %%
