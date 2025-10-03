import pickle

from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)


def merge_params(params_dict):
    """Merge two parameter dictionaries"""
    params = params_dict["default"].copy()
    for key in params_dict.keys():
        if key != "default":
            param_list = params_dict[key]["names"]
            for param in param_list:
                params[param] = params_dict[key]["params"][param]

    return params


def merge_men_and_women_params(path_dict, ungendered_model_name):
    """Merge two parameter dictionaries for men and women into one."""
    params_men = pickle.load(
        open(
            path_dict["struct_results"] + f"est_params_{ungendered_model_name}_men.pkl",
            "rb",
        )
    )
    params_women = pickle.load(
        open(
            path_dict["struct_results"]
            + f"est_params_{ungendered_model_name}_women.pkl",
            "rb",
        )
    )

    params = params_men.copy()
    for key in params.keys():
        if key.endswith("_women"):
            params[key] = params_women[key]
        elif "children" in key:
            params[key] = params_women[key]

    return params


# def new_to_current(path_dict):
#     params = pickle.load(
#         open(path_dict["struct_results"] + f"est_params_new.pkl", "rb")
#     )
#     params["mu_men"] = params["mu"]
#     params["mu_women"] = params["mu"]
#     params.pop("mu")
#
#     params["taste_shock_scale_men"] = params["lambda"]
#     params["taste_shock_scale_women"] = params["lambda"]
#     params.pop("lambda")
#
#     for s in ["men", "women"]:
#         for edu in ["low", "high"]:
#             for health in ["bad", "good"]:
#                 params[f"disutil_ft_work_{edu}_{health}_{s}"] = params[
#                     f"disutil_ft_work_{health}_{s}"
#                 ]
#
#             params[f"disutil_unemployed_{edu}_{s}"] = params[f"disutil_unemployed_{s}"]
#
#     for edu in ["low", "high"]:
#         for health in ["bad", "good"]:
#             params[f"disutil_pt_work_{edu}_{health}_women"] = params[
#                 f"disutil_pt_work_{health}_women"
#             ]
#
#     params_start = load_and_set_start_params(path_dict)
#     for param in [
#         "disability_logit_const",
#         "disability_logit_age",
#         "disability_logit_high_educ",
#     ]:
#         for append in ["men", "women"]:
#             params[f"{param}_{append}"] = params_start[f"{param}_{append}"]
#     return params
#
#
# def map_period_to_age(params):
#     """Map period to age for the params dictionary."""
#     params["job_finding_logit_const_women"] -= (
#         params["job_finding_logit_period_women"] * 30
#     )
#     params["job_finding_logit_age_women"] = params["job_finding_logit_period_women"]
#     params.pop("job_finding_logit_period_women")
#
#     params["job_finding_logit_const_men"] -= params["job_finding_logit_period_men"] * 30
#     params["job_finding_logit_age_men"] = params["job_finding_logit_period_men"]
#     params.pop("job_finding_logit_period_men")
#
#     params["disability_logit_const"] -= params["disability_logit_period"] * 30
#     params["disability_logit_age"] = params["disability_logit_period"]
#     params.pop("disability_logit_period")
#
#     return params
#
#
# def gender_separate_models(params):
#     params["taste_shock_scale_men"] = params["lambda"]
#     params["taste_shock_scale_women"] = params["lambda"]
#     params.pop("lambda")
#
#     params["disability_logit_const_men"] = params["disability_logit_const"]
#     params["disability_logit_const_women"] = params["disability_logit_const"]
#     params.pop("disability_logit_const")
#
#     params["disability_logit_age_men"] = params["disability_logit_age"]
#     params["disability_logit_age_women"] = params["disability_logit_age"]
#     params.pop("disability_logit_age")
#
#     params["disability_logit_high_educ_men"] = params["disability_logit_high_educ"]
#     params["disability_logit_high_educ_women"] = params["disability_logit_high_educ"]
#     params.pop("disability_logit_high_educ")
#
#     return params
