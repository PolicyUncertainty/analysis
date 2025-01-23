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
    # "taste_shock_scale_men",
    # "taste_shock_scale_women",
    "lambda",
    "job_finding_logit_const_men",
    "job_finding_logit_age_men",
    "job_finding_logit_high_educ_men",
    "job_finding_logit_const_women",
    "job_finding_logit_age_women",
    "job_finding_logit_high_educ_women",
]

# last_estimate = pkl.load(open(paths_dict["est_results"] + "est_params_pete.pkl", "rb"))
# pop_list = [
#     "job_finding_logit_const_men",
#     "job_finding_logit_age_men",
#     "job_finding_logit_high_educ_men",
#     "job_finding_logit_const_women",
#     "job_finding_logit_age_women",
#     "job_finding_logit_high_educ_women",
# ]
# for pop in pop_list:
#     last_estimate.pop(pop)
#
# last_estimate.pop("disutil_unemployed_men")
# last_estimate.pop("disutil_unemployed_women")

last_estimate = {
    "lambda": 0.4830439358496316,
    "disutil_unemployed_men": 1.3758751590380753,
    "disutil_ft_work_good_men": 0.41016191418706827,
    "disutil_ft_work_bad_men": 1.2889853108697904,
    "job_finding_logit_const_men": 0.6902693377274312,
    "job_finding_logit_age_men": -0.041307031292744316,
    "job_finding_logit_high_educ_men": -0.18309737235303156,
    "disutil_unemployed_women": 0.9171530794400352,
    "disutil_ft_work_good_women": 1.444255548168784,
    "disutil_ft_work_bad_women": 1.8148955972087175,
    "disutil_children_ft_work_low": 0.1982561152991237,
    "disutil_children_ft_work_high": 0.11820300337224642,
    "disutil_pt_work_good_women": 1.2438774961479997,
    "disutil_pt_work_bad_women": 1.681060500666308,
    "job_finding_logit_const_women": 0.7041484563879603,
    "job_finding_logit_age_women": -0.059030109271216456,
    "job_finding_logit_high_educ_women": 0.5956866132506432,
}

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append="both",
    slope_disutil_method=False,
    load_model=True,
    use_weights=True,
    last_estimate=last_estimate,
    save_results=False,
)
print(estimation_results)


# %%
