import os
import pickle
import pickle as pkl
import time

import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
from dcegm.likelihood import create_individual_likelihood_function_for_model
from dcegm.wealth_correction import adjust_observed_wealth
from estimation.struct_estimation.start_params.set_start_params import (
    load_and_set_start_params,
)
from model_code.specify_model import specify_model
from model_code.stochastic_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)


def estimate_model(path_dict, params_to_estimate_names, file_append, load_model):
    if not os.path.exists(path_dict["intermediate_est_data"]):
        os.makedirs(path_dict["intermediate_est_data"])

    start_params_all = load_and_set_start_params(path_dict)

    # # Assign start params from before
    # last_end = pkl.load(open(path_dict["est_results"] + "est_params.pkl", "rb"))
    #
    # start_params_all.update(last_end)
    start_params_all["bequest_scale"] = 1

    individual_likelihood, weights = create_ll_from_paths(
        start_params_all, path_dict, load_model
    )

    start_params = {name: start_params_all[name] for name in params_to_estimate_names}

    def individual_likelihood_print(params):
        start = time.time()
        ll_value_individual, model_solution = individual_likelihood(params)
        ll_value = jnp.dot(weights, ll_value_individual)
        save_iter_step(
            model_solution, ll_value, params, path_dict["intermediate_est_data"]
        )
        end = time.time()
        print("Likelihood evaluation took, ", end - start)
        print("Params, ", params, " with ll value, ", ll_value)
        return ll_value

    # Define for all parameters the bounds. We do not need to do that for those
    # not estimated. They will selected afterwards.
    lower_bounds_all = {
        "mu": 1e-12,
        "dis_util_ft_work_high": 1e-12,
        "dis_util_ft_work_low": 1e-12,
        "dis_util_pt_work_high": 1e-12,
        "dis_util_pt_work_low": 1e-12,
        "dis_util_unemployed_high": 1e-12,
        "dis_util_unemployed_low": 1e-12,
        "bequest_scale": 1e-12,
        "lambda": 1e-12,
        "job_finding_logit_const": -5,
        "job_finding_logit_age": -0.5,
        "job_finding_logit_high_educ": -5,
    }
    lower_bounds = {name: lower_bounds_all[name] for name in params_to_estimate_names}
    upper_bounds_all = {
        "mu": 2,
        "dis_util_ft_work_high": 5,
        "dis_util_ft_work_low": 5,
        "dis_util_pt_work_high": 5,
        "dis_util_pt_work_low": 5,
        "dis_util_unemployed_high": 5,
        "dis_util_unemployed_low": 5,
        "bequest_scale": 5,
        "lambda": 0.5,
        "job_finding_logit_const": 5,
        "job_finding_logit_age": 0.5,
        "job_finding_logit_high_educ": 5,
    }

    upper_bounds = {name: upper_bounds_all[name] for name in params_to_estimate_names}

    bounds = om.Bounds(lower=lower_bounds, upper=upper_bounds)
    result = om.minimize(
        fun=individual_likelihood_print,
        params=start_params,
        bounds=bounds,
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
    data_decision, states_dict = load_and_prep_data(
        path_dict, start_params_all, model, drop_retirees=True
    )

    # Create specs for unobserved states
    unobserved_state_specs = create_unobserved_state_specs(data_decision, model)

    # Create likelihood function
    individual_likelihood = create_individual_likelihood_function_for_model(
        model=model,
        observed_states=states_dict,
        observed_choices=data_decision["choice"].values,
        unobserved_state_specs=unobserved_state_specs,
        params_all=start_params_all,
        return_model_solution=True,
    )
    return individual_likelihood, data_decision["age_weights"].values


def create_unobserved_state_specs(data_decision, model):
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
    return unobserved_state_specs


def load_and_prep_data(path_dict, start_params, model, drop_retirees=True):
    # Load data
    data_decision = pd.read_pickle(path_dict["struct_est_sample"])
    # We need to filter observations in period 0 because of job offer weighting
    data_decision = data_decision[data_decision["period"] > 0]
    # Also already retired individuals hold no identification
    if drop_retirees:
        data_decision = data_decision[data_decision["lagged_choice"] != 0]

    data_decision["age"] = (
        data_decision["period"] + model["options"]["model_params"]["start_age"]
    )
    data_decision["age_bin"] = np.floor(data_decision["age"] / 10)
    data_decision.loc[data_decision["age_bin"] > 6, "age_bin"] = 6
    age_bin_av_size = data_decision.shape[0] / data_decision["age_bin"].nunique()
    data_decision.loc[:, "age_weights"] = 1.0
    data_decision.loc[:, "age_weights"] = age_bin_av_size / data_decision.groupby(
        "age_bin"
    )["age_weights"].transform("sum")

    # Transform experience
    max_init_exp = model["options"]["model_params"]["max_init_experience"]
    exp_denominator = data_decision["period"].values + max_init_exp
    data_decision["experience"] = data_decision["experience"] / exp_denominator

    # We can adjust wealth outside, as it does not depend on estimated parameters
    # (only on interest rate)
    # Now transform for dcegm
    states_dict = {
        name: data_decision[name].values
        for name in model["model_structure"]["discrete_states_names"]
    }
    states_dict["experience"] = data_decision["experience"].values
    states_dict["wealth"] = data_decision["wealth"].values

    adjusted_wealth = adjust_observed_wealth(
        observed_states_dict=states_dict,
        params=start_params,
        model=model,
    )
    data_decision["adjusted_wealth"] = adjusted_wealth
    states_dict["wealth"] = data_decision["adjusted_wealth"].values

    return data_decision, states_dict


def save_iter_step(model_sol, ll_value, params, logging_folder):
    saving_object = {"model_sol": model_sol, "ll_value": ll_value, "params": params}
    pkl.dump(saving_object, open(logging_folder + "solving_log.pkl", "wb"))
