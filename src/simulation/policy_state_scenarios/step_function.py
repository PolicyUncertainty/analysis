import jax
import jax.numpy as jnp
import numpy as np


def realized_policy_step_function(policy_state, period, lagged_choice, choice, options):
    """This function yields the probability distribution of the next period's policy
    state in the simulation. We employ a step function to follow the expected policy
    state path.

    If the person is already retired, the next policy state has to be the degenerate
    state.

    """

    # Check if the current period is a policy step period
    step_period = jnp.isin(period, options["policy_step_periods"])

    # Check if retirement is choosen
    retirement_bool = choice == 0
    # If retirement is choosen the transition vector is a zero vector with a one at the
    # current state and if we are in a step period and not retired then the transition
    # vector has probability 1 of increase in policy state. Retirement superseeds the
    # step period
    id_next_period = step_period * (policy_state + 1) + (1 - step_period) * policy_state

    id_next_period = (
        retirement_bool * policy_state + (1 - retirement_bool) * id_next_period
    )

    # If the individual is already retired, the policy state moves to (or stays in)
    # the degenrate state
    degenerate_state_id = jnp.array(options["n_policy_states"] - 1, dtype=jnp.uint8)
    already_retirement_bool = (lagged_choice == 0) | (
        policy_state == degenerate_state_id
    )

    id_next_period = jax.lax.select(
        already_retirement_bool, degenerate_state_id, id_next_period
    )

    # Now generate vector
    trans_vector = jnp.zeros(options["n_policy_states"])
    trans_vector = trans_vector.at[id_next_period].set(1)
    return trans_vector


def create_update_function_for_scale(path_dict, scale):
    # Load the estimates
    alpha_hat = np.loadtxt(path_dict["est_results"] + "exp_val_params.txt") * scale
    return create_update_function_for_slope(alpha_hat)


def create_update_function_for_slope(slope):
    def update_specs_for_step_function_with_slope(specs, path_dict):
        return update_specs_step_function_with_slope(specs, slope)

    return update_specs_for_step_function_with_slope


def update_specs_step_function_with_slope(specs, slope):
    # Generate policy state steps for individuals who start in 0. First calculate the
    # per year expected increase in policy state
    life_span = specs["end_age"] - specs["start_age"] + 1
    per_period_expec_increase = np.arange(life_span) * slope
    # Then round to the nearest value, which you can do by multiplying with the
    # inverse of the grid size. In the baseline case 1 / 0.25 = 4
    multiplier = 1 / specs["SRA_grid_size"]
    policy_state_ids = np.round(per_period_expec_increase * multiplier)
    specs["policy_step_periods"] = (
        np.where(policy_state_ids > np.roll(policy_state_ids, shift=1))[0] - 1
    )
    return specs
