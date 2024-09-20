import pickle
import pickle as pkl
import time

import estimagic as em
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from dcegm.likelihood import create_individual_likelihood_function_for_model
from dcegm.wealth_correction import adjust_observed_wealth
from model_code.specify_model import specify_model
from model_code.stochastic_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from specs.derive_specs import generate_derived_and_data_derived_specs


def estimate_model(path_dict, params_to_estimate_names, file_append, load_model):
    start_params_all = yaml.safe_load(open(path_dict["start_params"], "rb"))

    job_sep_params = create_job_offer_params_from_start(path_dict)
    start_params_all.update(job_sep_params)

    # Assign start params from before
    last_end = pkl.load(open(path_dict["est_results"] + "est_params.pkl", "rb"))
    start_params_all.update(last_end)

    # Spread out to edu types
    specs = generate_derived_and_data_derived_specs(path_dict)
    start_params_all["dis_util_work"] = (
        jnp.ones(specs["n_education_types"], dtype=jnp.float64)
        * start_params_all["dis_util_work"]
    )
    start_params_all["dis_util_unemployed"] = (
        jnp.ones(specs["n_education_types"], dtype=jnp.float64)
        * start_params_all["dis_util_unemployed"]
    )

    individual_likelihood = create_ll_from_paths(
        start_params_all, path_dict, load_model
    )

    start_params = {name: start_params_all[name] for name in params_to_estimate_names}

    def individual_likelihood_print(params):
        start = time.time()
        ll_value = individual_likelihood(params)[0]
        end = time.time()
        print("Likelihood evaluation took, ", end - start)
        print("Params, ", params, " with ll value, ", ll_value)
        return ll_value

    # Define for all parameters the bounds. We do not need to do that for those
    # not estimated. They will selected afterwards.
    lower_bounds_all = {
        "mu": 1e-12,
        "dis_util_work": jnp.array([1e-12, 1e-12], dtype=jnp.float64),
        "dis_util_unemployed": jnp.array([1e-12, 1e-12], dtype=jnp.float64),
        "bequest_scale": 1e-12,
        "lambda": 1e-12,
        "job_finding_logit_const": -5,
        "job_finding_logit_age": -0.5,
        "job_finding_logit_high_educ": -5,
    }
    lower_bounds = {name: lower_bounds_all[name] for name in params_to_estimate_names}
    upper_bounds_all = {
        "mu": 2,
        "dis_util_work": jnp.array([5, 5], dtype=jnp.float64),
        "dis_util_unemployed": jnp.array([5, 5], dtype=jnp.float64),
        "bequest_scale": 5,
        "lambda": 1,
        "job_finding_logit_const": 5,
        "job_finding_logit_age": 0.5,
        "job_finding_logit_high_educ": 5,
    }
    upper_bounds = {name: upper_bounds_all[name] for name in params_to_estimate_names}

    result = em.minimize(
        criterion=individual_likelihood_print,
        params=start_params,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        algorithm="scipy_lbfgsb",
        # logging="test_log.db",
        error_handling="continue",
    )
    pickle.dump(
        result, open(path_dict["est_results"] + f"em_result_{file_append}.pkl", "wb")
    )
    start_params_all.update(result.params)
    pickle.dump(
        start_params_all,
        open(path_dict["est_results"] + f"est_params_{file_append}.pkl", "wb"),
    )

    return result


def create_ll_from_paths(start_params_all, path_dict, load_model):
    model, params = specify_model(
        path_dict=path_dict,
        params=start_params_all,
        update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
        policy_state_trans_func=expected_SRA_probs_estimation,
        load_model=load_model,
    )

    # Load data
    data_decision, states_dict = load_and_prep_data(path_dict, start_params_all, model)

    def weight_func(**kwargs):
        # We need to weight the unobserved job offer state for each of its possible values
        # The weight function is called with job offer new beeing the unobserved state
        job_offer = kwargs["job_offer_new"]
        return model["model_funcs"]["processed_exog_funcs"]["job_offer"](**kwargs)[
            job_offer
        ]

    relevant_prev_period_state_choices_dict = {
        "period": data_decision["period"].values - 1,
        "education": data_decision["education"].values,
    }
    unobserved_state_specs = {
        "observed_bool": data_decision["full_observed_state"].values,
        "weight_func": weight_func,
        "states": ["job_offer"],
        "pre_period_states": relevant_prev_period_state_choices_dict,
        "pre_period_choices": data_decision["lagged_choice"].values,
    }

    # Create likelihood function
    individual_likelihood = create_individual_likelihood_function_for_model(
        model=model,
        observed_states=states_dict,
        observed_wealth=data_decision["adjusted_wealth"].values,
        observed_choices=data_decision["choice"].values,
        unobserved_state_specs=unobserved_state_specs,
        params_all=start_params_all,
        # weights=data_decision["age_weights"].values,
    )
    return individual_likelihood


def load_and_prep_data(path_dict, start_params, model, drop_retirees=True):
    # Load data
    data_decision = pd.read_pickle(path_dict["struct_est_sample"])
    # We need to filter observations in period 0 because of job offer weighting
    data_decision = data_decision[data_decision["period"] > 0]
    # Also already retired individuals hold no identification
    if drop_retirees:
        data_decision = data_decision[data_decision["lagged_choice"] != 2]

    # data_decision["age"] = (
    #     data_decision["period"] + model["options"]["model_params"]["start_age"]
    # )
    # data_decision["age_bin"] = np.floor(data_decision["age"] / 10)
    # data_decision.loc[data_decision["age_bin"] > 6, "age_bin"] = 6
    # age_bin_av_size = data_decision.shape[0] / data_decision["age_bin"].nunique()
    # data_decision.loc[:, "age_weights"] = 1.0
    # data_decision.loc[:, "age_weights"] = age_bin_av_size / data_decision.groupby(
    #     "age_bin"
    # )["age_weights"].transform("sum")

    # We can adjust wealth outside, as it does not depend on estimated parameters
    # (only on interest rate)
    # Now transform for dcegm
    states_dict = {
        name: data_decision[name].values
        for name in model["model_structure"]["state_space_names"]
    }

    adjusted_wealth = adjust_observed_wealth(
        observed_states_dict=states_dict,
        wealth=data_decision["wealth"].values,
        params=start_params,
        model=model,
    )
    data_decision["adjusted_wealth"] = adjusted_wealth
    return data_decision, states_dict


def create_job_offer_params_from_start(path_dict):
    struct_est_sample = pd.read_pickle(path_dict["struct_est_sample"])

    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

    logit_df = struct_est_sample[struct_est_sample["lagged_choice"] == 0][
        ["period", "education", "choice"]
    ].copy()
    logit_df["age"] = logit_df["period"] + specs["start_age"]

    # logit_df["above_49"] = 0
    # logit_df.loc[logit_df["age"] > 49, "above_49"] = 1

    logit_df = logit_df[logit_df["age"] < 65]
    logit_df = logit_df[logit_df["choice"] != 2]
    logit_df["intercept"] = 1

    logit_vars = [
        "intercept",
        "age",
        "education",
    ]

    logit_model = sm.Logit(logit_df["choice"], logit_df[logit_vars])
    logit_fitted = logit_model.fit()

    params = logit_fitted.params

    job_offer_params = {
        "job_finding_logit_const": params["intercept"],
        "job_finding_logit_age": params["age"],
        "job_finding_logit_high_educ": params["education"],
    }
    return job_offer_params
