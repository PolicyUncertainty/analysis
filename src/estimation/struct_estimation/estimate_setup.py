import os
import pickle
import pickle as pkl
import time

import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
import yaml
from dcegm.likelihood import create_individual_likelihood_function_for_model
from dcegm.wealth_correction import adjust_observed_wealth
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)
from estimation.struct_estimation.std_errors import calc_and_save_standard_errors
from model_code.specify_model import specify_model
from model_code.stochastic_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from model_code.unobserved_state_weighting import create_unobserved_state_specs
from specs.derive_specs import generate_derived_and_data_derived_specs


def estimate_model(
    path_dict,
    params_to_estimate_names,
    file_append,
    slope_disutil_method,
    load_model,
    use_weights=True,
    last_estimate=None,
    save_results=True,
):
    print_function = generate_print_func(params_to_estimate_names)
    # Load start params and bounds
    start_params_all = load_and_set_start_params(path_dict)

    print_function(last_estimate)

    # # Assign start params from before
    if last_estimate is not None:
        for key in last_estimate.keys():
            if key in ["sigma", "interest_rate", "beta"]:
                continue
            try:
                print(
                    f"Start params value of {key} was {start_params_all[key]} and is"
                    f"replaced by {last_estimate[key]}"
                )
            except:
                raise ValueError(f"Key {key} not found in start params.")
            start_params_all[key] = last_estimate[key]

    start_params = {name: start_params_all[name] for name in params_to_estimate_names}

    lower_bounds_all = yaml.safe_load(
        open(path_dict["start_params_and_bounds"] + "lower_bounds.yaml", "rb")
    )
    lower_bounds = {name: lower_bounds_all[name] for name in params_to_estimate_names}

    upper_bounds_all = yaml.safe_load(
        open(path_dict["start_params_and_bounds"] + "upper_bounds.yaml", "rb")
    )
    upper_bounds = {name: upper_bounds_all[name] for name in params_to_estimate_names}

    bounds = om.Bounds(lower=lower_bounds, upper=upper_bounds)

    # Initialize estimation class
    est_class = est_class_from_paths(
        path_dict=path_dict,
        start_params_all=start_params_all,
        slope_disutil_method=slope_disutil_method,
        print_function=print_function,
        file_append=file_append,
        load_model=load_model,
        use_weights=use_weights,
        save_results=save_results,
    )

    result = om.minimize(
        fun=est_class.crit_func,
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

    calc_and_save_standard_errors(
        path_dict=path_dict,
        ll_func=est_class.ll_func,
        final_params_all=start_params_all,
        final_params_est=result.params,
        params_to_estimate_names=params_to_estimate_names,
        weights=est_class.weights,
        file_append=file_append,
    )


class est_class_from_paths:
    def __init__(
        self,
        path_dict,
        start_params_all,
        slope_disutil_method,
        file_append,
        load_model,
        use_weights,
        print_function=None,
        save_results=True,
    ):
        self.iter_count = 0
        self.slope_disutil_method = slope_disutil_method
        self.save_results = save_results

        if print_function is None:
            self.print_function = lambda params: print("Params, ", pd.Series(params))
        else:
            self.print_function = print_function

        intermediate_est_data = (
            path_dict["intermediate_data"] + f"estimation_{file_append}/"
        )
        if not os.path.exists(intermediate_est_data):
            os.makedirs(intermediate_est_data)

        self.intermediate_est_data = intermediate_est_data

        model, params = specify_model(
            path_dict=path_dict,
            params=start_params_all,
            update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
            policy_state_trans_func=expected_SRA_probs_estimation,
            load_model=load_model,
            model_type="solution",
        )
        # Load data
        data_decision, states_dict = load_and_prep_data(
            path_dict, start_params_all, model, drop_retirees=True
        )

        if use_weights:
            self.weights = data_decision["age_weights"].values
            self.weight_sum = np.sum(self.weights)
        else:
            self.weights = np.ones(data_decision.shape[0])
            self.weight_sum = data_decision.shape[0]

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
        self.ll_func = individual_likelihood
        specs = generate_derived_and_data_derived_specs(path_dict)
        # self.pt_ratio_low = (
        #     specs["av_annual_hours_pt"][0] / specs["av_annual_hours_ft"][0]
        # )
        # self.pt_ratio_high = (
        #     specs["av_annual_hours_pt"][1] / specs["av_annual_hours_ft"][1]
        # )

    def crit_func(self, params):
        start = time.time()
        # if self.slope_disutil_method:
        #     params = update_according_to_slope_disutil(
        #         params, self.pt_ratio_low, self.pt_ratio_high
        #     )
        ll_value_individual, model_solution = self.ll_func(params)
        ll_value = jnp.dot(self.weights, ll_value_individual) / self.weight_sum
        if self.save_results:
            save_iter_step(
                model_solution,
                ll_value,
                params,
                self.intermediate_est_data,
                self.iter_count,
            )
        end = time.time()
        self.iter_count += 1
        self.print_function(params)
        print("Likelihood value: ", ll_value)
        print("Likelihood evaluation took, ", end - start)

        return ll_value


# def update_according_to_slope_disutil(params, pt_ratio_bad, pt_ratio_good):
#     """Use this function to entforce slope condition of disutility parameters."""
#     params["disutil_unemployed_bad"] = params["disutil_not_retired_low"]
#     params["disutil_pt_work_bad"] = (
#         params["disutil_not_retired_bad"]
#         + pt_ratio_bad * params["disutil_working_bad"]
#     )
#     params["disutil_ft_work_bad"] = (
#         params["disutil_not_retired_bad"] + params["disutil_working_bad"]
#     )
#
#     params["disutil_unemployed"] = params["disutil_not_retired_good"]
#     params["disutil_pt_work_good"] = (
#         params["disutil_not_retired_good"]
#         + pt_ratio_good * params["disutil_working_good"]
#     )
#     params["disutil_ft_work_good"] = (
#         params["disutil_not_retired_good"] + params["disutil_working_good"]
#     )
#     return params


def save_iter_step(model_sol, ll_value, params, logging_folder, iter_count):
    alternate_save_count = iter_count % 2
    saving_object = {"model_sol": model_sol, "ll_value": ll_value}
    pkl.dump(
        saving_object,
        open(logging_folder + f"solving_log_{alternate_save_count}.pkl", "wb"),
    )
    pkl.dump(params, open(logging_folder + f"params_{alternate_save_count}.pkl", "wb"))


def load_and_prep_data(path_dict, start_params, model, drop_retirees=True):
    specs = generate_derived_and_data_derived_specs(path_dict)
    # Load data
    data_decision = pd.read_pickle(path_dict["struct_est_sample"])
    # We need to filter observations in period 0 because of job offer weighting from last period
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
    states_dict["wealth"] = data_decision["wealth"].values / specs["wealth_unit"]

    adjusted_wealth = adjust_observed_wealth(
        observed_states_dict=states_dict,
        params=start_params,
        model=model,
    )
    data_decision["adjusted_wealth"] = adjusted_wealth
    states_dict["wealth"] = data_decision["adjusted_wealth"].values

    return data_decision, states_dict


def generate_print_func(params_to_estimate_names):
    men_params = get_gendered_params(params_to_estimate_names, "_men")
    women_params = get_gendered_params(params_to_estimate_names, "_women")

    for param in ["disutil_children_ft_work_low", "disutil_children_ft_work_high"]:
        if param in params_to_estimate_names:
            women_params["all"] += [param]
            women_params["full-time"] += [param]
    neutral_params = [
        param_name
        for param_name in params_to_estimate_names
        if param_name not in men_params["all"] + women_params["all"]
    ]
    men_params.pop("all")
    women_params.pop("all")

    taste_shock_params = [
        param_name
        for param_name in params_to_estimate_names
        if "taste_shock" in param_name
    ]

    def print_function(params):
        print("Gender neutral parameters:")
        for param_name in neutral_params:
            print(f"{param_name}: {params[param_name]}")
        # print("\nTaste shock parameters:")
        # for param_name in taste_shock_params:
        #     print(f"{param_name}: {params[param_name]}")
        print("\nMen model params are:")
        for group_name in men_params.keys():
            print(f"Group: {group_name}")
            for param in men_params[group_name]:
                if "disutil" in param:
                    print(
                        f"{param}: {params[param]} and in probability: {np.exp(-params[param])}"
                    )
                elif "job_finding" in param:
                    print(f"{param}: {params[param]}")
        print("\nParameters of the women model are:")
        for group_name in women_params.keys():
            print(f"Group: {group_name}")
            for param in women_params[group_name]:
                if "disutil" in param:
                    print(
                        f"{param}: {params[param]} and in probability: {np.exp(-params[param])}"
                    )
                elif "job_finding" in param:
                    print(f"{param}: {params[param]}")

    return print_function


def get_gendered_params(params_to_estimate_names, append):
    gender_params = [
        param_name for param_name in params_to_estimate_names if append in param_name
    ]

    disutil_params = [
        param_name for param_name in gender_params if "disutil_" in param_name
    ]
    disutil_params_pt_params = [
        param_name for param_name in disutil_params if "pt_work" in param_name
    ]
    disutil_params_ft_params = [
        param_name for param_name in disutil_params if "ft_work" in param_name
    ]
    job_finding_params = [
        param_name for param_name in gender_params if "job_finding_" in param_name
    ]

    # We do it this weird way for printing order
    params = {}
    if f"disutil_unemployed{append}" in disutil_params:
        params["unemployed"] = [f"disutil_unemployed{append}"]

    if len(disutil_params_ft_params) > 0:
        params["full-time"] = disutil_params_ft_params

    if len(disutil_params_pt_params) > 0:
        params["part-time"] = disutil_params_pt_params

    if len(job_finding_params) > 0:
        params["job_finding"] = job_finding_params

    # We drop these directly afterwards
    if len(gender_params) > 0:
        params["all"] = gender_params
    return params
