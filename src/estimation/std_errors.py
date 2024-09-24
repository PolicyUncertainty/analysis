import pickle

import estimagic as sm
import jax
import numpy as np
from estimation.estimate_setup import create_ll_from_paths
from jax.flatten_util import ravel_pytree

# Import jax and set jax to work with 64bit
jax.config.update("jax_enable_x64", True)

# %%
# Set paths of project
from set_paths import create_path_dict

path_dict = create_path_dict()


params = pickle.load(open(path_dict["est_results"] + "est_params_all.pkl", "rb"))

individual_likelihood, weights = create_ll_from_paths(
    params, path_dict, load_model=True
)
n_obs = weights.sum()
hessian_weights = np.sqrt(weights)[:, None]

params_to_estimate_names = [
    # "mu",
    "dis_util_work_high",
    "dis_util_work_low",
    "dis_util_unemployed_high",
    "dis_util_unemployed_low",
    "bequest_scale",
    # "lambda",
    "job_finding_logit_const",
    "job_finding_logit_age",
    "job_finding_logit_high_educ",
]

params_est = {name: params[name] for name in params_to_estimate_names}

unravel_func = ravel_pytree(params_est)[0]
contributions_base = individual_likelihood(params_est)

scores = np.zeros((contributions_base.shape[0], len(params_to_estimate_names)))
for param_id, param_name in enumerate(params_to_estimate_names):
    params_plus = params_est.copy()
    params_plus[param_name] += 1e-6
    contributions_plus = individual_likelihood(params_est)

    scores_param = (contributions_plus - contributions_base) / (1e-6)
    scores[:, param_id] = scores_param

weighted_scores = scores * hessian_weights
hessian = weighted_scores.T @ weighted_scores / n_obs
std_errors = np.sqrt(np.diag(np.linalg.inv(hessian) / n_obs))

pickle.dump(
    unravel_func(std_errors),
    open(path_dict["est_results"] + "std_errors_all.pkl", "wb"),
)
# array([0.00717759, 0.00455941, 0.02647084, 0.01402793, 0.26843421,
#        0.08585728, 0.00139075, 0.04709085])
