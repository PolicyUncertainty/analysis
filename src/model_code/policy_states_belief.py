import jax.numpy as jnp
import numpy as np
from scipy.stats import norm


def expected_SRA_probs_estimation(policy_state, choice, options):
    trans_mat = options["beliefs_trans_mat"]
    # Take the row of the transition matrix for expected policy change
    trans_vector_not_retired = jnp.take(trans_mat, policy_state, axis=0)
    # If already retired the transition vector is a zero vector with a one at the
    # current state
    n_exog_states = trans_mat.shape[0]
    no_policy_change = jnp.zeros(n_exog_states)
    no_policy_change = no_policy_change.at[policy_state].set(1)
    # Check if retired
    retirement_bool = choice == 2
    # Aggregate the two transition vectors
    trans_vector = (
        1 - retirement_bool
    ) * trans_vector_not_retired + retirement_bool * no_policy_change
    return trans_vector


def exp_ret_age_transition_matrix(specs, project_paths):
    # Load parameters
    alpha_hat = np.loadtxt(project_paths["est_results"] + "exp_val_params.txt")
    sigma_sq_hat = np.loadtxt(project_paths["est_results"] + "var_params.txt")

    # Read out specs
    step_size = specs["SRA_grid_size"]
    n_policy_states = specs["n_policy_states"]
    labels = specs["SRA_values_policy_states"]

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

    specs["beliefs_trans_mat"] = ret_age_exp_transition_matrix
    return specs
