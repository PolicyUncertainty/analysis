import os
import pickle
import pickle as pkl
import time

import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
import yaml

from estimation.struct_estimation.scripts.std_errors import (
    calc_and_save_standard_errors,
    calc_scores,
)
from model_code.specify_model import specify_model
from model_code.stochastic_processes.job_offers import (
    calc_job_finding_prob_men,
    calc_job_finding_prob_women,
    job_sep_probability,
)
from model_code.transform_data_from_model import (
    create_states_dict,
    load_scale_and_correct_data,
)
from model_code.unobserved_state_weighting import create_unobserved_state_specs
from specs.derive_specs import generate_derived_and_data_derived_specs


def estimate_model(
    path_dict,
    params_to_estimate_names,
    file_append,
    load_model,
    start_params_all,  #
    supply_jacobian=False,
    use_weights=True,
    last_estimate=None,
    save_results=True,
    print_men_examples=True,
    print_women_examples=False,
    use_observed_data=True,
    sim_data=None,
    sex_type="all",
    edu_type="all",
    util_type="add",
    scale_opt=False,
    multistart=False,
    slow_version=False,
):

    specs = generate_derived_and_data_derived_specs(path_dict)

    print_function = generate_print_func(
        params_to_estimate_names,
        specs,
        print_men_examples=print_men_examples,
        print_women_examples=print_women_examples,
    )

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
    print_function(start_params_all)

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
        specs=specs,
        start_params_all=start_params_all,
        print_function=print_function,
        file_append=file_append,
        load_model=load_model,
        use_weights=use_weights,
        save_results=save_results,
        print_men_examples=print_men_examples,
        print_women_examples=print_women_examples,
        use_observed_data=use_observed_data,
        sim_data=sim_data,
        sex_type=sex_type,
        edu_type=edu_type,
        slow_version=slow_version,
        util_type=util_type,
    )

    if supply_jacobian:
        add_kwargs = {"jac": est_class.jacobian_func}
    else:
        add_kwargs = {}

    if scale_opt:
        add_kwargs["scaling"] = om.ScalingOptions(method="bounds")

    if multistart:
        add_kwargs["multistart"] = om.MultistartOptions(n_samples=5, seed=0)

    result = om.minimize(
        fun=est_class.crit_func,
        params=start_params,
        bounds=bounds,
        algorithm="scipy_lbfgsb",
        # logging="test_log.db",
        error_handling="continue",
        **add_kwargs,
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
    return result, start_params_all


class est_class_from_paths:
    def __init__(
        self,
        path_dict,
        specs,
        start_params_all,
        file_append,
        load_model,
        use_weights,
        print_function=None,
        print_men_examples=True,
        print_women_examples=True,
        save_results=True,
        sim_data=None,
        use_observed_data=True,
        slow_version=False,
        sex_type="all",
        edu_type="all",
        util_type="add",
    ):
        self.iter_count = 0
        self.save_results = save_results
        self.print_men_examples = print_men_examples
        self.print_women_examples = print_women_examples
        self.start_params_all = start_params_all

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
        self.specs = specs

        model = specify_model(
            path_dict=path_dict,
            specs=specs,
            subj_unc=True,
            custom_resolution_age=None,
            load_model=load_model,
            sim_specs=None,
            sex_type=sex_type,
            edu_type=edu_type,
            util_type=util_type,
        )

        if use_observed_data:
            # Load data
            data_decision = load_and_prep_data_estimation(
                path_dict=path_dict, model_class=model
            )
        else:
            if not isinstance(sim_data, pd.DataFrame):
                raise ValueError("If not using observed data, sim_data must be given.")
            data_decision = sim_data.copy()

        data_decision = filter_data_by_type(
            df=data_decision, sex_type=sex_type, edu_type=edu_type
        )
        # Already retired individuals hold no identification
        data_decision = data_decision[data_decision["lagged_choice"] != 0]
        # Create states dict
        states_dict = create_states_dict(data_decision, model_class=model)

        if use_weights:
            raise ValueError("Weights currently not supported.")
            # self.weights = data_decision["age_weights"].values
            # self.weight_sum = np.sum(self.weights)
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
            slow_version=slow_version,
        )
        self.ll_func = individual_likelihood

    def crit_func(self, params):
        start = time.time()
        ll_value_individual = self.ll_func(params)
        ll_value = jnp.dot(self.weights, ll_value_individual) / self.weight_sum
        full_params = self.start_params_all.copy()
        full_params.update(params)
        if self.save_results:
            alternate_save_count = self.iter_count % 2
            pkl.dump(
                full_params,
                open(
                    self.intermediate_est_data + f"params_{alternate_save_count}.pkl",
                    "wb",
                ),
            )

        end = time.time()
        self.iter_count += 1
        self.print_function(full_params)

        print("Likelihood value: ", ll_value)
        print("Likelihood evaluation took, ", end - start)

        return ll_value

    def jacobian_func(self, params):
        """Calculate the jacobian of the likelihood function."""

        scores = calc_scores(
            ll_func=self.ll_func,
            est_params=params,
            params_to_estimate_names=params.keys(),
        )
        return self.weights @ scores / self.weight_sum


def load_and_prep_data_estimation(path_dict, model_class):

    data_decision = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_class
    )

    # data_decision["age_bin"] = np.floor(data_decision["age"] / 10)
    # data_decision.loc[data_decision["age_bin"] > 6, "age_bin"] = 6
    # age_bin_av_size = data_decision.shape[0] / data_decision["age_bin"].nunique()
    # data_decision.loc[:, "age_weights"] = 1.0
    # data_decision.loc[:, "age_weights"] = age_bin_av_size / data_decision.groupby(
    #     "age_bin"
    # )["age_weights"].transform("sum")

    return data_decision


def generate_print_func(
    params_to_estimate_names, specs, print_men_examples=True, print_women_examples=True
):
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
    gender_labels = ["Men", "Women"]

    def print_function(params):
        print("Gender neutral parameters:")
        for param_name in neutral_params:
            print(f"{param_name}: {params[param_name]}")
        # print("\nTaste shock parameters:")
        # for param_name in taste_shock_params:
        #     print(f"{param_name}: {params[param_name]}")
        for i, gender_params in enumerate([men_params, women_params]):
            print(f"{gender_labels[i]} parameters are:")
            for group_name in gender_params.keys():
                print(f"Group: {group_name}")
                for param_name in gender_params[group_name]:
                    print(
                        f"{param_name}: {params[param_name]}",
                        flush=True,
                    )

        if print_men_examples:
            job_offer_prob_60_high_good = calc_job_finding_prob_men(
                params=params,
                education=1,
                good_health=1,
                age=60,
            )
            print(
                f"Job offer prob for 60 year old high educated men in good health: {job_offer_prob_60_high_good}",
                flush=True,
            )
            job_sep = job_sep_probability(
                params=params,
                policy_state=8,
                education=0,
                sex=0,
                age=66,
                good_health=1,
                model_specs=specs,
            )
            print(
                f"Job separation prob for 66 year old low educated men next year at SRA 67: {job_sep}",
                flush=True,
            )

        if print_women_examples:
            job_offer_prob_60_high_good = calc_job_finding_prob_women(
                params=params,
                education=1,
                good_health=1,
                age=60,
            )
            print(
                f"Job offer prob for 60 year old high educated women in good health: {job_offer_prob_60_high_good}",
                flush=True,
            )
            job_sep = job_sep_probability(
                params=params,
                policy_state=8,
                education=0,
                sex=1,
                age=66,
                good_health=1,
                model_specs=specs,
            )
            print(
                f"Job separation prob for 66 year old low educated women next year at SRA 67: {job_sep}",
                flush=True,
            )

    return print_function


def get_gendered_params(params_to_estimate_names, append):
    gender_params = [
        param_name for param_name in params_to_estimate_names if append in param_name
    ]

    job_finding_params = [
        param_name for param_name in gender_params if "job_finding_" in param_name
    ]

    disability_params = [
        param_name for param_name in gender_params if "disability" in param_name
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
    disutil_params_partner = [
        param_name for param_name in disutil_params if "partner" in param_name
    ]
    SRA_firing_params = [
        param_name for param_name in gender_params if "SRA_firing" in param_name
    ]

    # We do it this weird way for printing order
    params = {}
    # Assign all gender params. This will be dropped afterwards
    if len(gender_params) > 0:
        params["all"] = gender_params

    # Assign group params
    if len(disutil_unemployed_params) > 0:
        params["unemployed"] = disutil_unemployed_params

    if len(disutil_params_ft_params) > 0:
        params["full-time"] = disutil_params_ft_params

    if len(disutil_params_pt_params) > 0:
        params["part-time"] = disutil_params_pt_params

    if len(job_finding_params) > 0:
        params["job_finding"] = job_finding_params

    if len(disability_params) > 0:
        params["disability_logit"] = disability_params

    if len(disutil_params_partner) > 0:
        params["partner"] = disutil_params_partner

    if len(SRA_firing_params) > 0:
        params["SRA_firing"] = SRA_firing_params

    other_params = []
    for param in gender_params:
        if (
            param
            not in job_finding_params
            + disability_params
            + disutil_params
            + SRA_firing_params
            + disutil_params_partner
        ):
            other_params += [param]

    if len(other_params) > 0:
        params["other_params"] = other_params

    return params


def filter_data_by_type(df, sex_type, edu_type):
    if sex_type == "men":
        df = df[df["sex"] == 0]
    elif sex_type == "women":
        df = df[df["sex"] == 1]
    elif sex_type == "all":
        pass
    else:
        raise ValueError("sex_type not recognized.")

    if edu_type == "low":
        df = df[df["education"] == 0]
    elif edu_type == "high":
        df = df[df["education"] == 1]
    elif edu_type == "all":
        pass
    else:
        raise ValueError("edu_type not recognized.")

    return df
