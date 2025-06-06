import os
import pickle
import pickle as pkl
import time

import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
import yaml
from dcegm.asset_correction import adjust_observed_assets

from estimation.struct_estimation.scripts.std_errors import (
    calc_and_save_standard_errors,
)
from model_code.specify_model import specify_model
from model_code.unobserved_state_weighting import create_unobserved_state_specs
from process_data.structural_sample_scripts.create_structural_est_sample import (
    CORE_TYPE_DICT,
)
from specs.derive_specs import generate_derived_and_data_derived_specs


def estimate_model(
    path_dict,
    params_to_estimate_names,
    file_append,
    load_model,
    start_params_all,
    use_weights=True,
    last_estimate=None,
    save_results=True,
):
    print_function = generate_print_func(params_to_estimate_names)

    # # Assign start params from before
    if last_estimate is not None:
        print_function(last_estimate)

        for key in start_params_all:
            try:
                print(
                    f"Start params value of {key} was {start_params_all[key]} and is "
                    f"replaced by {last_estimate[key]}",
                    flush=True,
                )
            except:
                raise ValueError(f"Key {key} not found in last_estimate.")
            start_params_all[key] = last_estimate[key]

    start_params = {name: start_params_all[name] for name in params_to_estimate_names}
    print_function(start_params)

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
        result, open(path_dict["struct_results"] + f"em_result_{file_append}.pkl", "wb")
    )
    start_params_all.update(result.params)

    pickle.dump(
        start_params_all,
        open(path_dict["struct_results"] + f"est_params_{file_append}.pkl", "wb"),
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
    return result


class est_class_from_paths:
    def __init__(
        self,
        path_dict,
        start_params_all,
        file_append,
        load_model,
        use_weights,
        print_function=None,
        save_results=True,
    ):
        self.iter_count = 0
        self.save_results = save_results

        if print_function is None:
            self.print_function = lambda params: print("Params, ", pd.Series(params))
        else:
            self.print_function = print_function

        intermediate_est_data = (
            path_dict["intermediate_data"] + f"estimation_{file_append}/"
        )
        if save_results:
            if not os.path.exists(intermediate_est_data):
                os.makedirs(intermediate_est_data)

        self.intermediate_est_data = intermediate_est_data

        model = specify_model(
            path_dict=path_dict,
            subj_unc=True,
            custom_resolution_age=None,
            load_model=load_model,
            sim_specs=None,
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
        unobserved_state_specs = create_unobserved_state_specs(
            data_decision=data_decision
        )

        # Create likelihood function
        individual_likelihood = model.create_experimental_ll_func(
            observed_states=states_dict,
            observed_choices=data_decision["choice"].values,
            unobserved_state_specs=unobserved_state_specs,
            params_all=start_params_all,
            return_model_solution=False,
        )
        self.ll_func = individual_likelihood

    def crit_func(self, params):
        start = time.time()
        ll_value_individual = self.ll_func(params)
        ll_value = jnp.dot(self.weights, ll_value_individual) / self.weight_sum
        if self.save_results:
            alternate_save_count = self.iter_count % 2
            pkl.dump(
                params,
                open(
                    self.intermediate_est_data + f"params_{alternate_save_count}.pkl",
                    "wb",
                ),
            )
        end = time.time()
        self.iter_count += 1
        self.print_function(params)
        print("Likelihood value: ", ll_value)
        print("Likelihood evaluation took, ", end - start)

        return ll_value


def load_and_prep_data(path_dict, start_params, model_class, drop_retirees=True):
    specs = generate_derived_and_data_derived_specs(path_dict)
    # Load data
    data_decision = pd.read_csv(path_dict["struct_est_sample"])
    data_decision = data_decision.astype(CORE_TYPE_DICT)

    # Also already retired individuals hold no identification
    if drop_retirees:
        data_decision = data_decision[data_decision["lagged_choice"] != 0]

    model_specs = model_class.model_specs

    data_decision["age"] = data_decision["period"] + model_specs["start_age"]
    data_decision["age_bin"] = np.floor(data_decision["age"] / 10)
    data_decision.loc[data_decision["age_bin"] > 6, "age_bin"] = 6
    age_bin_av_size = data_decision.shape[0] / data_decision["age_bin"].nunique()
    data_decision.loc[:, "age_weights"] = 1.0
    data_decision.loc[:, "age_weights"] = age_bin_av_size / data_decision.groupby(
        "age_bin"
    )["age_weights"].transform("sum")

    # Transform experience
    max_init_exp = model_specs["max_exp_diffs_per_period"][
        data_decision["period"].values
    ]
    exp_denominator = data_decision["period"].values + max_init_exp
    data_decision["experience"] = data_decision["experience"] / exp_denominator

    # We can adjust wealth outside, as it does not depend on estimated parameters
    # (only on interest rate)
    # Now transform for dcegm
    states_dict = {
        name: data_decision[name].values
        for name in model_class.model_structure["discrete_states_names"]
    }
    states_dict["experience"] = data_decision["experience"].values
    states_dict["assets_begin_of_period"] = (
        data_decision["wealth"].values / specs["wealth_unit"]
    )

    assets_begin_of_period = adjust_observed_assets(
        observed_states_dict=states_dict,
        params=start_params,
        model_class=model_class,
    )
    data_decision["assets_begin_of_period"] = assets_begin_of_period
    states_dict["assets_begin_of_period"] = assets_begin_of_period

    return data_decision, states_dict


def generate_print_func(params_to_estimate_names):
    men_params = get_gendered_params(params_to_estimate_names, "_men")
    women_params = get_gendered_params(params_to_estimate_names, "_women")
    for param_dict in [men_params, women_params]:
        if "all" not in param_dict.keys():
            param_dict["all"] = []

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

    # taste_shock_params = [
    #     param_name
    #     for param_name in params_to_estimate_names
    #     if "taste_shock" in param_name
    # ]

    def print_function(params):
        print("Gender neutral parameters:")
        for param_name in neutral_params:
            print(f"{param_name}: {params[param_name]}")
        # print("\nTaste shock parameters:")
        # for param_name in taste_shock_params:
        #     print(f"{param_name}: {params[param_name]}")
        print("\nMen model params are:")
        for gender_params in [men_params, women_params]:
            for group_name in gender_params.keys():
                print(f"Group: {group_name}")
                for param_name in gender_params[group_name]:
                    if "disutil" in param_name:
                        print(
                            f"{param_name}: {params[param_name]} and in probability: {np.exp(-params[param_name])}"
                        )
                    else:
                        print(f"{param_name}: {params[param_name]}")

    return print_function


def get_gendered_params(params_to_estimate_names, append):
    gender_params = [
        param_name for param_name in params_to_estimate_names if append in param_name
    ]

    disutil_params = [
        param_name for param_name in gender_params if "disutil_" in param_name
    ]
    disutil_unemployed_params = [
        param_name for param_name in disutil_params if "unemployed" in param_name
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
    taste_shock_scale_params = [
        param_name for param_name in gender_params if "taste_shock" in param_name
    ]

    disability_params = [
        param_name for param_name in gender_params if "disability" in param_name
    ]
    # We do it this weird way for printing order
    params = {}
    if len(disutil_unemployed_params) > 0:
        params["unemployed"] = disutil_unemployed_params

    if len(disutil_params_ft_params) > 0:
        params["full-time"] = disutil_params_ft_params

    if len(disutil_params_pt_params) > 0:
        params["part-time"] = disutil_params_pt_params

    if len(job_finding_params) > 0:
        params["job_finding"] = job_finding_params

    if len(taste_shock_scale_params) > 0:
        params["taste_shock"] = taste_shock_scale_params

    if len(disability_params) > 0:
        params["disability"] = disability_params

    # We drop these directly afterwards
    if len(gender_params) > 0:
        params["all"] = gender_params
    return params
