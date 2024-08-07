import pickle
import time

import estimagic as em
import pandas as pd
import yaml
from dcegm.likelihood import create_individual_likelihood_function_for_model
from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat
from model_code.specify_model import specify_model


def estimate_model(path_dict, load_model):
    start_params_all = yaml.safe_load(open(path_dict["start_params"], "rb"))

    individual_likelihood = create_ll_from_paths(
        start_params_all, path_dict, load_model
    )

    def individual_likelihood_print(params):
        start = time.time()
        ll_value = individual_likelihood(params)[0]
        end = time.time()
        print("Likelihood evaluation took, ", end - start)
        print("Params, ", params, " with ll value, ", ll_value)
        return ll_value

    params_to_estimate_names = [
        # "mu",
        "dis_util_work",
        "dis_util_unemployed",
        # "bequest_scale",
        # "lambda",
        "job_finding_logit_const",
        "job_finding_logit_age",
        "job_finding_logit_age_squ",
        "job_finding_logit_high_educ",
    ]
    start_params = {name: start_params_all[name] for name in params_to_estimate_names}

    lower_bounds = {
        # "mu": 1e-12,
        "dis_util_work": 1e-12,
        "dis_util_unemployed": 1e-12,
        # "bequest_scale": 1e-12,
        # "lambda": 1e-12,
        "job_finding_logit_const": -10,
        "job_finding_logit_age": -10,
        "job_finding_logit_age_squ": -10,
        "job_finding_logit_high_educ": -10,
    }
    upper_bounds = {
        # "mu": 5,
        "dis_util_work": 50,
        "dis_util_unemployed": 50,
        # "bequest_scale": 20,
        # "lambda": 1,
        "job_finding_logit_const": 10,
        "job_finding_logit_age": 10,
        "job_finding_logit_age_squ": 10,
        "job_finding_logit_high_educ": 10,
    }

    result = em.minimize(
        criterion=individual_likelihood_print,
        params=start_params,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        algorithm="scipy_lbfgsb",
        logging="test_log.db",
        error_handling="continue",
    )
    pickle.dump(result, open(path_dict["est_results"] + "em_result.pkl", "wb"))
    start_params_all.update(result.params)
    pickle.dump(
        start_params_all, open(path_dict["est_results"] + "est_params.pkl", "wb")
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
    data_decision = pd.read_pickle(
        path_dict["intermediate_data"] + "structural_estimation_sample.pkl"
    )
    data_decision = data_decision[data_decision["lagged_choice"] != 2]
    data_decision["wealth"] = data_decision["wealth"].clip(lower=1e-16)
    # Now transform for dcegm
    states_dict = {
        name: data_decision[name].values
        for name in model["model_structure"]["state_space_names"]
    }

    # Create likelihood function
    individual_likelihood = create_individual_likelihood_function_for_model(
        model=model,
        observed_states=states_dict,
        observed_wealth=data_decision["wealth"].values,
        observed_choices=data_decision["choice"].values,
        params_all=start_params_all,
    )
    return individual_likelihood
