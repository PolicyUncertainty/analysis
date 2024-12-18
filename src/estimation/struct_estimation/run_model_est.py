# Set paths of project
import pickle as pkl

import pandas as pd
from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)

from estimation.struct_estimation.estimate_setup import estimate_model

params_to_estimate_names = [
    # "mu",
    "dis_util_unemployed_high",
    "dis_util_pt_work_high",
    "dis_util_ft_work_high",
    "dis_util_unemployed_low",
    "dis_util_ft_work_low",
    "dis_util_pt_work_low",
    # "dis_util_not_retired_low",
    # "dis_util_working_low",
    # "dis_util_not_retired_high",
    # "dis_util_working_high",
    # "bequest_scale",
    # "lambda",
    "job_finding_logit_const",
    "job_finding_logit_age",
    "job_finding_logit_high_educ",
]
pop_list = [
    "dis_util_not_retired_low",
    "dis_util_working_low",
    "dis_util_not_retired_high",
    "dis_util_working_high",
]
last_estimate = pkl.load(
    open(paths_dict["est_results"] + "est_params_cet_par.pkl", "rb")
)
# for pop in pop_list:
#     last_estimate.pop(pop)
print("Restart estimation at: ", pd.Series(last_estimate))

estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append="cet_par",
    slope_disutil_method=False,
    load_model=False,
    last_estimate=last_estimate,
    save_results=False,
)
print(estimation_results)


# %%
