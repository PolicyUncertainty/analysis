import time

import numpy as np
from jax.flatten_util import ravel_pytree


class est_class:
    def __init__(
        self,
        log_like,
        start_params_to_est,
        counts,
        atol_reusing=1e-14,
    ):
        """Log like has as first value the weighted log like."""
        self._crit_func = log_like
        self.base_value = np.ones_like(counts)
        self.unravel_func = ravel_pytree(start_params_to_est)[1]
        self.crit_eval_params = ravel_pytree(start_params_to_est)[0]
        self.atol_reusing = atol_reusing
        self.counts = counts
        self.n_obs = counts.sum()
        self.obs_weights = counts / self.n_obs
        self.hessian_weights = np.sqrt(counts)[:, None]
        self.f_count = 0
        self.score_count = 0

    def crit_func(self, params_vec):
        # Translate to array
        start = time.time()
        params_dict = self.unravel_func(params_vec)

        # Calc ll value and save
        individual_values = self._crit_func(params_dict)
        self.base_value = individual_values
        self.crit_eval_params = params_vec
        end = time.time()
        print("Likelihood evaluation took, ", end - start)
        ll_val = (individual_values * self.obs_weights).sum()
        print("Params, ", params_dict, " with ll value, ", ll_val)
        self.f_count += 1
        return ll_val

    def jac_func(self, params_vec):
        params_dict = self.unravel_func(params_vec)
        if np.allclose(params_vec, self.crit_eval_params, atol=self.atol_reusing):
            pass
        else:
            self.base_value = self._crit_func(params_dict)
            self.crit_eval_params = params_vec

        self.score_eval_params = params_vec
        self.scores = numerical_scores(
            self.base_value,
            params_vec,
            self.unravel_func,
            self._crit_func,
            epsilon=1e-6,
        )
        self.score_count += 1
        return self.obs_weights @ self.scores

    def hessian_prox(self, params_vec):
        params_dict = self.unravel_func(params_vec)
        if np.allclose(params_vec, self.score_eval_params, atol=self.atol_reusing):
            pass
        else:
            if np.allclose(params_vec, self.crit_eval_params, atol=self.atol_reusing):
                pass
            else:
                self.base_value = self._crit_func(params_dict)
                self.crit_eval_params = params_vec
            self.scores = numerical_scores(
                self.base_value,
                params_vec,
                self.unravel_func,
                self._crit_func,
                epsilon=1e-6,
            )
            self.score_eval_params = params_vec

        weighted_score = self.scores * self.hessian_weights
        return np.dot(weighted_score.T, weighted_score) / self.n_obs


def numerical_scores(baseline_value, params_vec, unravel_func, crit_func, epsilon=1e-6):
    """Calculate the numerical scores for the given parameters."""
    scores = np.zeros((baseline_value.shape[0], params_vec.shape[0]), dtype=float)
    for i in range(len(params_vec)):
        params_vec_int = params_vec.copy()
        params_vec_int[i] += epsilon
        params_dict = unravel_func(params_vec_int)
        scores[:, i] = (crit_func(params_dict) - baseline_value) / epsilon
    return scores
