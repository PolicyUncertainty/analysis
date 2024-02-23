import pickle

import estimagic as em
import pandas as pd
from dcegm.likelihood import create_individual_likelihood_function_for_model
from estimation.tools import generate_model_to_estimate
from estimation.tools import process_data_for_dcegm
from model_code.budget_equation import create_savings_grid


def estimate_model(project_paths, load_model):
    start_params_all = {
        # Utility parameters
        "mu": 0.5,
        "dis_util_work": 4.0,
        "dis_util_unemployed": 1.0,
        "bequest_scale": 2.0,
        # Taste and income shock scale
        "lambda": 1.0,
        # Interest rate and discount factor
        "interest_rate": 0.03,
        "beta": 0.95,
    }

    model, options, start_params_all = generate_model_to_estimate(
        project_paths=project_paths,
        start_params_all=start_params_all,
        load_model=load_model,
    )
    data_decision = pd.read_pickle(
        project_paths["intermediate_data"] + "decision_data.pkl"
    )
    data_decision = data_decision[data_decision["lagged_choice"] != 2]
    data_dict = process_data_for_dcegm(data_decision, model["state_space_names"])

    savings_grid = create_savings_grid()

    # Create likelihood function
    individual_likelihood = create_individual_likelihood_function_for_model(
        model=model,
        options=options,
        observed_states=data_dict["states"],
        observed_wealth=data_dict["wealth"],
        observed_choices=data_dict["choices"],
        exog_savings_grid=savings_grid,
        params_all=start_params_all,
    )

    params_to_estimate_names = [
        # "mu",
        "dis_util_work",
        "dis_util_unemployed",
        "bequest_scale",
        # "lambda",
        # "sigma",
    ]
    start_params = {name: start_params_all[name] for name in params_to_estimate_names}

    def individual_likelihood_print(params):
        ll_value = individual_likelihood(params)
        print("Params, ", params, " with ll value, ", ll_value)
        return ll_value

    lower_bounds = {
        "dis_util_work": 1e-12,
        "dis_util_unemployed": 1e-12,
        "bequest_scale": 1e-12,
        # "lambda": 1e-12,
    }
    upper_bounds = {
        "dis_util_work": 100,
        "dis_util_unemployed": 100,
        "bequest_scale": 10,
        # "lambda": 100,
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
    pickle.dump(result, open(project_paths["est_results"] + "em_result.pkl", "wb"))
