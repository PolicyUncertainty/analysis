import pickle

import pandas as pd
from set_paths import create_path_dict

path_dict = create_path_dict()
# Import specs
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)

model_name = "new"
params_util = [
    "disutil_unemployed_men",
    "disutil_ft_work_good_men",
    "disutil_ft_work_bad_men",
    "disutil_pt_work_good_women",
    "disutil_ft_work_good_women",
    "disutil_ft_work_bad_women",
    "disutil_pt_work_bad_women",
]

params_all = pickle.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)
std_errors = pickle.load(
    open(path_dict["struct_results"] + f"std_errors_{model_name}.pkl", "rb")
)

param_appends_sex = ["men", "women"]
param_appends_choice = ["unemployed", "pt_work", "ft_work"]
param_appends_health = ["bad", "good"]
util_df = pd.DataFrame()
for sex_var, sex_label in enumerate(specs["sex_labels"]):
    param_append = param_appends_sex[sex_var]
    for choice_var, work_label in enumerate(specs["choice_labels"]):
        # Use choice var definition of model. Continue for retirement.
        if choice_var == 0:
            continue
        if (sex_var == 0) & (choice_var == 2):
            continue
        elif choice_var == 1:
            param_name_pre = f"disutil_unemployed_{param_append}"
            util_df.loc[
                f"{work_label}", sex_label
            ] = f"{params_all[param_name_pre]:.4f}"
            util_df.loc[
                f"{work_label}_se", sex_label
            ] = f"({std_errors[param_name_pre]:.4f})"
        else:
            for health_var in specs["alive_health_vars"]:
                health_param = param_appends_health[health_var]
                work_param = param_appends_choice[choice_var - 1]
                param_name_pre = f"disutil_{work_param}_{health_param}_{param_append}"

                health_label = specs["health_labels"][health_var]
                util_df.loc[
                    f"{work_label}; {health_label}", sex_label
                ] = f"{params_all[param_name_pre]:.4f}"
                util_df.loc[
                    f"{work_label}; {health_label}_se", sex_label
                ] = f"({std_errors[param_name_pre]:.4f})"

edu_param_labels = ["low", "high"]
for edu_var, edu_label in enumerate(specs["education_labels"]):
    param_name_pre = f"disutil_children_ft_work_{edu_param_labels[edu_var]}"
    if param_name_pre in params_all.keys():
        util_df.loc[
            f"Children; Full-time; {edu_label}", "Women"
        ] = f"{params_all[param_name_pre]:.4f}"
        util_df.loc[
            f"Children; Full-time; {edu_label}_se", "Women"
        ] = f"({std_errors['disutil_children_ft_work_low']:.4f})"

taste_shock_scale_name = "lambda"
util_df.loc["Taste shock scale", "Women"] = f"{params_all[taste_shock_scale_name]:.4f}"
util_df.loc[
    "Taste shock scale_se", "Women"
] = f"({std_errors[taste_shock_scale_name]:.4f})"
util_df.loc["Taste shock scale", "Men"] = f"{params_all[taste_shock_scale_name]:.4f}"
util_df.loc[
    "Taste shock scale_se", "Men"
] = f"({std_errors[taste_shock_scale_name]:.4f})"


def transform_df_to_latex_body(param_df):
    latex_body = param_df.to_latex()
    # Remove unwanted LaTeX commands
    latex_by_lines = latex_body.splitlines()[4:-2]
    # Split each string in list by "&"
    latex_by_cells = [line.split("&") for line in latex_by_lines]
    # Replace all sells which have a string _se by empty cell container {}
    latex_body_new = ""
    for line in latex_by_cells:
        new_line = [cell if "_se" not in cell else "{}" for cell in line]
        # Replace nans with empty cell container
        new_line = [cell if "NaN" not in cell else "{}" for cell in new_line]
        # now join the line back together
        latex_body_new += " & ".join(new_line) + "\n "
    return latex_body_new


latex_body = transform_df_to_latex_body(util_df)
with open(path_dict["tables"] + "util_params.tex", "w") as f:
    f.write(latex_body)


row_names = {
    "job_finding_logit_const": "Constant",
    "job_finding_logit_age": "Age",
    "job_finding_logit_high_educ": "High education",
}
job_offer_df = pd.DataFrame()

for param_append in param_appends_sex:
    for param_name_pre, row_name in row_names.items():
        param_name = f"{param_name_pre}_{param_append}"
        job_offer_df.loc[row_name, param_append] = f"{params_all[param_name]:.4f}"
        job_offer_df.loc[
            row_name + "_se", param_append
        ] = f"({std_errors[param_name]:.4f})"


latex_body = transform_df_to_latex_body(job_offer_df)
with open(path_dict["tables"] + "job_offer.tex", "w") as f:
    f.write(latex_body)
