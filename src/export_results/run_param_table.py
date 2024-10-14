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

params = pickle.load(open(path_dict["est_results"] + "est_params.pkl", "rb"))
std_errors = pickle.load(open(path_dict["est_results"] + "std_errors_all.pkl", "rb"))

util_df = pd.DataFrame()
for choice_label in ["work", "unemployed"]:
    for edu_label in ["low", "high"]:
        non_edu_name = f"dis_util_{choice_label}"
        param_name = f"dis_util_{choice_label}_{edu_label}"
        util_df.loc[non_edu_name, edu_label] = f"{params[param_name]:.4f}"
        util_df.loc[non_edu_name + "_se", edu_label] = f"({std_errors[param_name]:.4f})"

util_df.to_latex(path_dict["tables"] + "util.tex")

row_names = {
    "job_finding_logit_const": "Constant",
    "job_finding_logit_age": "Age",
    "job_finding_logit_high_educ": "High education",
}
job_offer_df = pd.DataFrame()

for param_name, row_name in row_names.items():
    job_offer_df.loc[row_name, "value"] = f"{params[param_name]:.4f}"
    job_offer_df.loc[row_name + "_se", "value"] = f"({std_errors[param_name]:.4f})"

job_offer_df.to_latex(path_dict["tables"] + "job_offer.tex")
