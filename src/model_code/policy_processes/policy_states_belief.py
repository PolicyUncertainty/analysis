import jax.lax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm


def expected_SRA_with_resolution(
    period, policy_state, choice, lagged_choice, model_specs
):
    trans_mat = model_specs["policy_states_trans_mat"]
    # Take the row of the transition matrix for expected policy change
    trans_vector_not_retired = jnp.take(trans_mat, policy_state, axis=0)

    # If fresh retired, you stay one more year in the same policy state
    fresh_retired = (choice == 0) & (lagged_choice != 0)
    # With resolution, the policy state also does not change after resolution period
    resolution_period = model_specs["resolution_age"] - model_specs["start_age"]
    past_resolution = period >= resolution_period
    no_change = fresh_retired | past_resolution
    trans_vector = select_no_policy_change(
        no_change_bool=no_change,
        n_policy_states=model_specs["n_policy_states"],
        current_policy_state=policy_state,
        trans_vector_policy_change=trans_vector_not_retired,
    )

    trans_vector = check_for_longer_retirement_and_degenerate_vector(
        choice=choice,
        lagged_choice=lagged_choice,
        degenerate_probs=trans_mat[-1, :],
        trans_vector=trans_vector,
    )

    return trans_vector


def select_no_policy_change(
    no_change_bool, n_policy_states, current_policy_state, trans_vector_policy_change
):
    no_policy_change = jnp.zeros(n_policy_states, dtype=float)
    no_policy_change = no_policy_change.at[current_policy_state].set(1)

    # Aggregate the two transition vectors
    trans_vector = jax.lax.select(
        no_change_bool, on_true=no_policy_change, on_false=trans_vector_policy_change
    )
    return trans_vector


def check_for_longer_retirement_and_degenerate_vector(
    choice,
    lagged_choice,
    degenerate_probs,
    trans_vector,
):
    # Check if already longer retired, then take transition
    # probabilities for degenerate state
    already_retired = (choice == 0) & (lagged_choice == 0)
    # Set to degenerate if already retired
    trans_vector = jax.lax.select(
        already_retired, on_true=degenerate_probs, on_false=trans_vector
    )
    return trans_vector


def update_specs_exp_ret_age_trans_mat(specs, path_dict):
    # Load parameters
    alpha_hat = np.loadtxt(path_dict["est_results"] + "exp_val_params.txt")
    sigma_sq_hat = np.loadtxt(path_dict["est_results"] + "var_params.txt")

    # Read out specs
    step_size = specs["SRA_grid_size"]
    n_beliefs_states = specs["n_policy_states"] - 1
    labels = specs["SRA_values_policy_states"]

    # create matrix of zeros and row/column labels
    ret_age_exp_transition_matrix = np.zeros((n_beliefs_states, n_beliefs_states))

    # fill in the matrix with the transition probabilities from the normal CDF
    for i in range(n_beliefs_states):
        for j in range(n_beliefs_states):
            delta = labels[j] - labels[i]
            # if the column is min ret age, p = CDF(delta - step_size/2)
            if j == 0:
                ret_age_exp_transition_matrix[i, j] = norm.cdf(
                    delta + step_size / 2, loc=alpha_hat, scale=sigma_sq_hat**0.5
                )
            # if the column is max ret age, p = 1 - CDF(delta + step_size/2)
            elif j == n_beliefs_states - 1:
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

    # Append zeros and a one in the down right corner for the degenerate state
    policy_states_trans_mat = np.hstack(
        (ret_age_exp_transition_matrix, np.zeros(n_beliefs_states).reshape(-1, 1))
    )
    last_row = np.zeros(n_beliefs_states + 1, dtype=float)
    last_row[-1] = 1
    policy_states_trans_mat = np.vstack((policy_states_trans_mat, last_row))
    specs["policy_states_trans_mat"] = policy_states_trans_mat
    return specs


#
# def expected_SRA_probs_estimation_without_resolution(policy_state, choice, lagged_choice, model_specs):
#     trans_mat = model_specs["policy_states_trans_mat"]
#     # Take the row of the transition matrix for expected policy change
#     trans_vector_not_retired = jnp.take(trans_mat, policy_state, axis=0)
#
#     # If fresh retired, you stay one more year in the same policy state
#     fresh_retired = (choice == 0) & (lagged_choice != 0)
#     trans_vector = select_no_policy_change(
#         no_change_bool=fresh_retired,
#         n_policy_states=model_specs["n_policy_states"],
#         current_policy_state=policy_state,
#         trans_vector_policy_change=trans_vector_not_retired,
#     )
#
#     trans_vector = check_for_longer_retirement_and_degenerate_vector(
#         choice=choice,
#         lagged_choice=lagged_choice,
#         degenerate_probs=trans_mat[-1, :],
#         trans_vector=trans_vector,
#     )
#
#     return trans_vector
