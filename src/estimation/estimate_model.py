import pickle
from pathlib import Path

import pandas as pd

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
file_dir_path = str(Path(__file__).resolve().parents[0]) + "/"
import sys
import jax
import estimagic as em
from estimation.tools_estimation import prepare_estimation

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")
project_paths = {
    "project_path": analysis_path,
    "model_path": file_dir_path,
}

data_decision = pd.read_pickle(analysis_path + "output/decision_data.pkl")

start_params_all = {
    # Utility parameters
    "mu": 0.5,
    "dis_util_work": 4.0,
    "dis_util_unemployed": 1.0,
    "bequest_scale": 2.0,
    # Taste and income shock scale
    "lambda": 1.0,
    "sigma": 1.0,
    # Interest rate and discount factor
    "interest_rate": 0.03,
    "beta": 0.95,
}


params_to_estimate_names = [
    # "mu",
    "dis_util_work",
    "dis_util_unemployed",
    "bequest_scale",
    "lambda",
    # "sigma",
]
start_params = {name: start_params_all[name] for name in params_to_estimate_names}


lower_bounds = {
    "dis_util_work": 1e-12,
    "dis_util_unemployed": 1e-12,
    "bequest_scale": 1e-12,
    "lambda": 1e-12,
}
upper_bounds = {
    "dis_util_work": 100,
    "dis_util_unemployed": 100,
    "bequest_scale": 10,
    "lambda": 100,
}

individual_likelihood = prepare_estimation(
    data_decision=data_decision,
    project_paths=project_paths,
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
pickle.dump(result, open(file_dir_path + "res.pkl", "wb"))

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
