import pickle

import pandas as pd
from set_paths import create_path_dict

path_dict = create_path_dict()

params_util = [
    # "mu",
    "dis_util_work_high",
    "dis_util_work_low",
    "dis_util_unemployed_high",
    "dis_util_unemployed_low",
    "bequest_scale",
    # "lambda",
]

params_job = [
    "job_finding_logit_const",
    "job_finding_logit_age",
    "job_finding_logit_high_educ",
]

params = pickle.load(open(path_dict["est_results"] + "est_params_all.pkl", "rb"))
std_errors = pickle.load(open(path_dict["est_results"] + "std_errors_all.pkl", "rb"))

util_df = pd.DataFrame()
