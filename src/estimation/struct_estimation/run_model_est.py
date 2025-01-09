# Set paths of project
import pickle as pkl

import pandas as pd
from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)

from estimation.struct_estimation.estimate_setup import estimate_model

params_to_estimate_names = [
    # "mu",
    "disutil_unemployed_men",
    "disutil_pt_work_good_men",
    "disutil_ft_work_good_men",
    "disutil_ft_work_bad_men",
    "disutil_pt_work_bad_men",
    "disutil_unemployed_women",
    "disutil_pt_work_good_women",
    "disutil_ft_work_good_women",
    "disutil_ft_work_bad_women",
    "disutil_pt_work_bad_women",
    # "disutil_not_retired_bad",
    # "disutil_working_bad",
    # "disutil_not_retired_good",
    # "disutil_working_good",
    # "bequest_scale",
    "lambda",
    "job_finding_logit_const_men",
    "job_finding_logit_age_men",
    "job_finding_logit_high_educ_men",
    "job_finding_logit_const_men",
    "job_finding_logit_age_men",
    "job_finding_logit_high_educ_men",
]
#
# last_estimate = pkl.load(open(paths_dict["est_results"] + "est_params_pete.pkl", "rb"))
# last_estimate["disutil_ft_work_high"]
# breakpoint()
# print("Restart estimation at: ", pd.Series(last_estimate))

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append="pete",
    slope_disutil_method=False,
    load_model=False,
    last_estimate=None,
    save_results=False,
)
print(estimation_results)


# %%
