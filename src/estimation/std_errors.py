import pickle

import estimagic as sm
import jax
import numpy as np
from estimation.estimate_setup import create_ll_func_from_path

# Import jax and set jax to work with 64bit
jax.config.update("jax_enable_x64", True)

# %%
# Set paths of project
from set_paths import create_path_dict

path_dict = create_path_dict()


params = pickle.load(open(path_dict["est_results"] + "est_params.pkl", "rb"))

individual_likelihood = create_ll_func_from_path(
    start_params_all=params, path_dict=path_dict, load_model=True
)

ll_value_base, contributions_base = individual_likelihood(params)

params_work_1 = params.copy()
params_work_1["dis_util_work"] += 1e-6
_, contributions_work_1 = individual_likelihood(params_work_1)

params_unemployed_1 = params.copy()
params_unemployed_1["dis_util_unemployed"] += 1e-6
_, contributions_unemployed_1 = individual_likelihood(params_unemployed_1)

score_work = (contributions_work_1 - contributions_base) / (1e-6)
score_unemployed = (contributions_unemployed_1 - contributions_base) / (1e-6)

scores = np.zeros((score_work.shape[0], 2))
scores[:, 0] = score_work
scores[:, 1] = score_unemployed

hessian = scores.T @ scores
std_errors = np.sqrt(np.diag(np.linalg.inv(hessian)))

pickle.dump(std_errors, open(path_dict["est_results"] + "std_errors.pkl", "wb"))
