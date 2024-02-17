import time
from pathlib import Path

import numpy as np
import pandas as pd

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
file_dir_path = str(Path(__file__).resolve().parents[0]) + "/"
import sys
import yaml
from jax.flatten_util import ravel_pytree
import pickle
import scipy.optimize as opt
import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")


data_decision = pd.read_pickle(analysis_path + "output/decision_data.pkl")

# Retirees don't have any choice and therefore no information
data_decision = data_decision[data_decision["lagged_choice"] != 2]

# Load data specs
from derive_specs import generate_derived_and_data_derived_specs

start = time.time()
project_paths = {
    "project_path": analysis_path,
}
project_specs = yaml.safe_load(open(analysis_path + "src/spec.yaml"))
project_specs = generate_derived_and_data_derived_specs(
    project_specs, project_paths, load_data=True
)
from model_code.specify_model import specify_model

model, options = specify_model(project_specs, load_model=True)
print("Model specified.")
# Prepare data for estimation
oberved_states_dict = {
    name: data_decision[name].values for name in model["state_space_names"]
}
observed_wealth = data_decision["wealth"].values
observed_choices = data_decision["choice"].values

# Load start parameters
start_params_all = yaml.safe_load(open(file_dir_path + "start_params.yaml"))
start_params_all["sigma"] = project_specs["income_shock_scale"]
# Specifiy savings wealth grid
savings_grid = np.arange(start=0, stop=100, step=0.5)
# Create likelihood function
from dcegm.likelihood import create_individual_likelihood_function_for_model

individual_likelihood = create_individual_likelihood_function_for_model(
    model=model,
    options=options,
    observed_states=oberved_states_dict,
    observed_wealth=observed_wealth,
    observed_choices=observed_choices,
    exog_savings_grid=savings_grid,
    params_all=start_params_all,
)

params_to_estimate_names = [
    # "mu",
    "dis_util_work",
    "dis_util_unemployed",
    "bequest_scale",
    "lambda",
    # "sigma",
]
start_params = {name: start_params_all[name] for name in params_to_estimate_names}

# Create likelihood function for estimation
params_start_vec, unravel_func = ravel_pytree(start_params)


def individual_likelihood_vec(params_vec):
    params = unravel_func(params_vec)
    ll_value = individual_likelihood(params)
    print("Params, ", params, " with ll value, ", ll_value)
    return ll_value


result = opt.minimize(
    individual_likelihood_vec,
    params_start_vec,
    bounds=[(1e-12, 100), (1e-12, 100), (1e-12, 100), (1e-12, 10)],
    method="L-BFGS-B",
)
pickle.dump(result, open(file_dir_path + "res.pkl", "wb"))

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
