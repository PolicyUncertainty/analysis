# Set paths of project
from set_paths import create_path_dict


paths_dict = create_path_dict(define_user=False)

from estimation.struct_estimation.estimate_setup import estimate_model

params_to_estimate_names = [
    # "mu",
    "dis_util_ft_work_high",
    "dis_util_ft_work_low",
    "dis_util_pt_work_high",
    "dis_util_pt_work_low",
    "dis_util_unemployed_high",
    "dis_util_unemployed_low",
    # "bequest_scale",
    "lambda",
    "job_finding_logit_const",
    "job_finding_logit_age",
    "job_finding_logit_high_educ",
]

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append="new",
    load_model=True,
)
print(estimation_results)


# %%
