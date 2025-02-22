# Set paths of project
import pickle as pkl

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)

from estimation.struct_estimation.scripts.estimate_setup import estimate_model

params_to_estimate_names = [
    # "mu",
    # Men Full-time - 4 parameters
    "disutil_ft_work_high_good_men",
    "disutil_ft_work_high_bad_men",
    "disutil_ft_work_low_good_men",
    "disutil_ft_work_low_bad_men",
    # Women Full-time - 4 parameters
    "disutil_ft_work_high_good_women",
    "disutil_ft_work_high_bad_women",
    "disutil_ft_work_low_good_women",
    "disutil_ft_work_low_bad_women",
    # Women Part-time - 4 parameters
    "disutil_pt_work_high_good_women",
    "disutil_pt_work_high_bad_women",
    "disutil_pt_work_low_good_women",
    "disutil_pt_work_low_bad_women",
    # Men unemployment - 2 parameters
    "disutil_unemployed_high_men",
    "disutil_unemployed_low_men",
    # Women Unemployment - 2 parameters
    "disutil_unemployed_high_women",
    "disutil_unemployed_low_women",
    # Children - 2 parameters
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
model_name = "partner_est"

last_estimate = pkl.load(
    open(paths_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)
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
    file_append=model_name,
    slope_disutil_method=False,
    load_model=False,
    use_weights=False,
    last_estimate=last_estimate,
    save_results=False,
)
print(estimation_results)


# %%
