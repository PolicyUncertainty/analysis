# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")

from set_paths import create_path_dict

do_first_step = input("Do first step estimation? (y/n): ") == "y"
paths_dict = create_path_dict(analysis_path, define_user=do_first_step)

# Import jax and set jax to work with 64bit
import jax

jax.config.update("jax_enable_x64", True)

# Import the rest
import pickle
import pandas as pd
import estimagic as em
from estimation.tools import create_likelihood


if do_first_step:
    # Estimate policy expectation
    from estimation.first_step_estimation.est_SRA_random_walk import (
        estimate_expected_SRA_variance,
        est_expected_SRA,
    )

    data_expectation = pd.read_pickle(
        paths_dict["intermediate_data"] + "policy_expect_data.pkl"
    )
    est_expected_SRA(paths_dict, data_expectation)
    estimate_expected_SRA_variance(paths_dict, data_expectation)

    # Estimate wage parameter
    from estimation.first_step_estimation.est_wage_equation import (
        estimate_wage_parameters,
    )

    estimate_wage_parameters(paths_dict)


data_decision = pd.read_pickle(paths_dict["intermediate_data"] + "decision_data.pkl")

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


params_to_estimate_names = [
    # "mu",
    "dis_util_work",
    "dis_util_unemployed",
    "bequest_scale",
    # "lambda",
    # "sigma",
]
start_params = {name: start_params_all[name] for name in params_to_estimate_names}


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

individual_likelihood = create_likelihood(
    data_decision=data_decision,
    project_paths=paths_dict,
    start_params_all=start_params_all,
    load_model=True,
)


def individual_likelihood_print(params):
    ll_value = individual_likelihood(params)
    print("Params, ", params, " with ll value, ", ll_value)
    return ll_value


result = em.minimize(
    criterion=individual_likelihood_print,
    params=start_params,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    algorithm="scipy_lbfgsb",
    logging="test_log.db",
    error_handling="continue",
)
pickle.dump(result, open(paths_dict["est_results"] + "res.pkl", "wb"))

#
# # Create likelihood function for estimation
# params_start_vec, unravel_func = ravel_pytree(start_params)
#
# bounds = [
# (1e-12, 100),
# (1e-12, 100),
# (1e-12, 100),
# (1e-12, 10)
# ]
# def individual_likelihood_vec(params_vec):
#     params = unravel_func(params_vec)
#     ll_value = individual_likelihood(params)
#     print("Params, ", params, " with ll value, ", ll_value)
#     return ll_value
#
# result = opt.minimize(
#     individual_likelihood_vec,
#     params_start_vec,
#     bounds=bounds,
#     method="L-BFGS-B",
# )
#


# from dcegm.solve import get_solve_func_for_model
# solve_func = get_solve_func_for_model(
#     model=model,exog_savings_grid=savings_grid, options=options
# )
# # start_params_all["bequest_scale"] = -48.0408919
# # start_params_all["dis_util_unemployed"] = -63.12383726
# # start_params_all["dis_util_work"] = -65.21426645
# #
# # value, policy_left, policy_right, endog_grid = solve_func(start_params_all)
# # breakpoint()


# past_prep = time.time()
# print(f"Preparation took {past_prep - start} seconds.")
# ll = individual_likelihood(start_params)
# first = time.time()
# print(f"First call took {first - past_prep} seconds.")
# ll = individual_likelihood(start_params)
# second = time.time()
# print(f"Second call took {second - first} seconds.")
