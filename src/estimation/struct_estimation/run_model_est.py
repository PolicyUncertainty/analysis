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

# last_estimate = {
#     "disutil_pt_work_good_women": 1.1613709241290286,
#     "disutil_ft_work_good_women": 1.420552617943543,
#     "disutil_ft_work_bad_women": 1.6456309911389186,
#     "disutil_pt_work_bad_women": 1.46455074176406,
#     "disutil_unemployed_women": 0.9001132529936234,
#     "disutil_children_ft_work_low": 0.1474776294937899,
#     "disutil_children_ft_work_high": 0.11002761203750037,
#     "lambda": 0.38401162033036007,
#     "job_finding_logit_const_women": 0.6743770385955012,
#     "job_finding_logit_age_women": -0.057388399968934925,
#     "job_finding_logit_high_educ_women": 0.6849652095064536,
#     "disutil_unemployed_men": 1.4660151006261883,
#     "disutil_ft_work_good_men": 0.6255058659669396,
#     "disutil_ft_work_bad_men": 1.315274144191485,
#     "job_finding_logit_const_men": 0.7212956269894059,
#     "job_finding_logit_age_men": -0.03850897252072163,
#     "job_finding_logit_high_educ_men": 0.1668213335038858,
# }

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append="both",
    slope_disutil_method=False,
    load_model=False,
    use_weights=True,
    last_estimate=None,
    save_results=False,
)
print(estimation_results)


# %%
