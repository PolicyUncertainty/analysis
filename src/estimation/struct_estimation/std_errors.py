import pickle

import jax
import numpy as np
from jax.flatten_util import ravel_pytree

# Import jax and set jax to work with 64bit
jax.config.update("jax_enable_x64", True)


def calc_and_save_standard_errors(
    path_dict,
    ll_func,
    final_params_all,
    final_params_est,
    params_to_estimate_names,
    weights,
    file_append,
):
    # Estimate standard errors
    contributions_base, _ = ll_func(final_params_all)

    unravel_func = ravel_pytree(final_params_est)[1]
    n_obs = weights.sum()
    hessian_weights = np.sqrt(weights)[:, None]

    eps = 1e-6
    scores = np.zeros((contributions_base.shape[0], len(params_to_estimate_names)))
    for param_id, param_name in enumerate(params_to_estimate_names):
        print(param_id)
        params_plus = final_params_all.copy()
        params_plus[param_name] += eps
        contributions_plus, _ = ll_func(params_plus)

        scores_param = (contributions_plus - contributions_base) / eps
        scores[:, param_id] = scores_param

    weighted_scores = scores * hessian_weights
    hessian = weighted_scores.T @ weighted_scores / n_obs
    std_errors = np.sqrt(np.diag(np.linalg.inv(hessian) / n_obs))

    pickle.dump(
        std_errors,
        open(path_dict["est_results"] + f"std_errors_{file_append}_raw.pkl", "wb"),
    )

    pickle.dump(
        unravel_func(std_errors),
        open(path_dict["est_results"] + f"std_errors_{file_append}.pkl", "wb"),
    )
