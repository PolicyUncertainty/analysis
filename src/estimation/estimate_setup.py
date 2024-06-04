import pickle
import time

import estimagic as em
from estimation.specify_likelihood import create_ll_from_paths


def estimate_model(path_dict, load_model):
    start_params_all = {
        # Utility parameters
        "mu": 0.8,
        "dis_util_work": 4.0,
        "dis_util_unemployed": 1.0,
        "bequest_scale": 1.3,
        # Taste and income shock scale
        "lambda": 1,
        # Interest rate and discount factor
        "interest_rate": 0.03,
        "beta": 0.97,
    }

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
    ]
    start_params = {name: start_params_all[name] for name in params_to_estimate_names}

    lower_bounds = {
        # "mu": 1e-12,
        "dis_util_work": 1e-12,
        "dis_util_unemployed": 1e-12,
        # "bequest_scale": 1e-12,
        # "lambda": 1e-12,
    }
    upper_bounds = {
        # "mu": 5,
        "dis_util_work": 50,
        "dis_util_unemployed": 50,
        # "bequest_scale": 20,
        # "lambda": 1,
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
