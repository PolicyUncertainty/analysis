import jax.numpy as jnp
import numpy as np
from scipy.stats import norm


def expected_SRA_probs(policy_state, options):
    trans_mat = options["beliefs_trans_mat"]
    trans_vector = jnp.take(trans_mat, policy_state, axis=0)
    return trans_vector


def exp_ret_age_transition_matrix(options, alpha_hat, sigma_sq_hat):
    step_size = options["SRA_grid_size"]
    n_policy_states = options["n_policy_states"]
    labels = options["SRA_values_policy_states"]

    # create matrix of zeros and row/column labels
    ret_age_exp_transition_matrix = np.zeros((n_policy_states, n_policy_states))

    # fill in the matrix with the transition probabilities from the normal CDF
    for i in range(n_policy_states):
        for j in range(n_policy_states):
            delta = labels[j] - labels[i]
            # if the column is min ret age, p = CDF(delta - step_size/2)
            if j == 0:
                ret_age_exp_transition_matrix[i, j] = norm.cdf(
                    delta + step_size / 2, loc=alpha_hat, scale=sigma_sq_hat**0.5
                )
            # if the column is max ret age, p = 1 - CDF(delta + step_size/2)
            elif j == n_policy_states - 1:
                ret_age_exp_transition_matrix[i, j] = 1 - norm.cdf(
                    delta - step_size / 2, loc=alpha_hat, scale=sigma_sq_hat**0.5
                )
            # otherwise, p = CDF(delta + step_size/2) - CDF(delta - step_size/2)
            else:
                ret_age_exp_transition_matrix[i, j] = norm.cdf(
                    delta + step_size / 2, loc=alpha_hat, scale=sigma_sq_hat**0.5
                ) - norm.cdf(
                    delta - step_size / 2, loc=alpha_hat, scale=sigma_sq_hat**0.5
                )

    return ret_age_exp_transition_matrix
