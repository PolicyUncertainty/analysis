import pickle

from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)


def new_to_current(path_dict):
    params = pickle.load(
        open(path_dict["struct_results"] + f"est_params_new.pkl", "rb")
    )
    params["mu_men"] = params["mu"]
    params["mu_women"] = params["mu"]
    params["job_finding_logit_period_men"] = params["job_finding_logit_age_men"]
    params["job_finding_logit_period_women"] = params["job_finding_logit_age_women"]
    params["job_finding_logit_const_men"] += params["job_finding_logit_age_men"] * 30
    params["job_finding_logit_const_women"] += (
        params["job_finding_logit_age_women"] * 30
    )

    for s in ["men", "women"]:
        for edu in ["low", "high"]:
            for health in ["bad", "good"]:
                params[f"disutil_ft_work_{edu}_{health}_{s}"] = params[
                    f"disutil_ft_work_{health}_{s}"
                ]

            params[f"disutil_unemployed_{edu}_{s}"] = params[f"disutil_unemployed_{s}"]

    for edu in ["low", "high"]:
        for health in ["bad", "good"]:
            params[f"disutil_pt_work_{edu}_{health}_women"] = params[
                f"disutil_pt_work_{health}_women"
            ]

    params_start = load_and_set_start_params(path_dict)
    params["disability_logit_const"] = params_start["disability_logit_const"]
    params["disability_logit_period"] = params_start["disability_logit_period"]
    params["disability_logit_high_educ"] = params_start["disability_logit_high_educ"]
    return params


def map_period_to_age(params):
    """Map period to age for the params dictionary."""
    params["job_finding_logit_const_women"] -= (
        params["job_finding_logit_period_women"] * 30
    )
    params["job_finding_logit_age_women"] = params["job_finding_logit_period_women"]
    params.pop("job_finding_logit_period_women")

    params["job_finding_logit_const_men"] -= params["job_finding_logit_period_men"] * 30
    params["job_finding_logit_age_men"] = params["job_finding_logit_period_men"]
    params.pop("job_finding_logit_period_men")

    params["disability_logit_const"] -= params["disability_logit_period"] * 30
    params["disability_logit_age"] = params["disability_logit_period"]
    params.pop("disability_logit_period")

    return params
