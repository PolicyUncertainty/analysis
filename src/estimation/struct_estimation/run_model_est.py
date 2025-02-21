# Set paths of project
from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)

from estimation.struct_estimation.scripts.estimate_setup import estimate_model

params_to_estimate_names = [
    # "mu",
    "disutil_ft_work_high_good_men",
    "disutil_ft_work_high_bad_men",
    "disutil_ft_work_low_good_men",
    "disutil_ft_work_low_bad_men",
    "disutil_ft_work_high_good_women",
    "disutil_ft_work_high_bad_women",
    "disutil_ft_work_low_good_women",
    "disutil_ft_work_low_bad_women",
    "disutil_pt_work_high_good_women",
    "disutil_pt_work_high_bad_women",
    "disutil_pt_work_low_good_women",
    "disutil_pt_work_low_bad_women",
    "disutil_unemployed_high_women",
    "disutil_unemployed_low_women",
    "disutil_children_ft_work_low",
    "disutil_children_ft_work_high",
    "lambda",
    "job_finding_logit_const_men",
    "job_finding_logit_age_men",
    "job_finding_logit_high_educ_men",
    "job_finding_logit_const_women",
    "job_finding_logit_age_women",
    "job_finding_logit_high_educ_women",
]

# last_estimate = pkl.load(open(paths_dict["struct_results"] + "est_params_pete.pkl", "rb"))
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

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append="partner_est",
    slope_disutil_method=False,
    load_model=False,
    use_weights=False,
    last_estimate=None,
    save_results=False,
)
print(estimation_results)


# %%
