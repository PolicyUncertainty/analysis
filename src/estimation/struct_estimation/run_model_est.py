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
    "lambda": 0.4944967815263844,
    "disutil_unemployed_men": 1.3828464645020635,
    "disutil_ft_work_good_men": 0.41631902366821383,
    "disutil_ft_work_bad_men": 1.2834272651304854,
    "job_finding_logit_const_men": 0.6889208090678306,
    "job_finding_logit_age_men": -0.04158755153234817,
    "job_finding_logit_high_educ_men": -0.1846405005412273,
    "disutil_unemployed_women": 0.9289395092867999,
    "disutil_ft_work_good_women": 1.4385501582869862,
    "disutil_ft_work_bad_women": 1.8218907759136247,
    "disutil_children_ft_work_low": 0.19135671006901858,
    "disutil_children_ft_work_high": 0.12283091684623981,
    "disutil_pt_work_good_women": 1.251966881444964,
    "disutil_pt_work_bad_women": 1.6824745473129654,
    "job_finding_logit_const_women": 0.7092057402172162,
    "job_finding_logit_age_women": -0.058648617500717336,
    "job_finding_logit_high_educ_women": 0.5967457796021962,
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
