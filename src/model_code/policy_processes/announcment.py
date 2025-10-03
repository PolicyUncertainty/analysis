import jax
import jax.numpy as jnp
import numpy as np


def announce_policy_state(policy_state, period, lagged_choice, model_specs):
    """This function implements an announcement process, where some policy can be
    announced at a some announcement period.

    The policy state is then updated to the announced policy state.

    """

    # Check if current period is announcement period. Otherwise policy_state stays
    bool_announcement = period == model_specs["announcement_period"]
    id_next_period = model_specs[
        "announced_policy_state"
    ] * bool_announcement + policy_state * (1 - bool_announcement)

    # If the individual is already retired, the policy state moves to (or stays in)
    # the degenrate state
    degenerate_state_id = jnp.array(model_specs["n_policy_states"] - 1, dtype=jnp.uint8)
    already_retirement_bool = (lagged_choice == 0) | (
        policy_state == degenerate_state_id
    )

    id_next_period = jax.lax.select(
        already_retirement_bool, degenerate_state_id, id_next_period
    )

    # Now generate vector
    trans_vector = jnp.zeros(model_specs["n_policy_states"])
    trans_vector = trans_vector.at[id_next_period].set(1)
    return trans_vector


def update_specs_for_policy_announcement(specs, announcement_age, announced_SRA):
    # Transform announcement age to period. We substract 1, such that agents are in the
    # announced SRA in announcment age. Therefore in age - 1 the transition has to be with certainty
    # to this policy state.
    announcement_period = announcement_age - specs["start_age"] - 1
    # Update specs
    specs["announcement_period"] = announcement_period

    # Transform announced_SRA to policy state
    announced_policy_state = int(
        (announced_SRA - specs["min_SRA"]) / specs["SRA_grid_size"]
    )
    specs["announced_policy_state"] = announced_policy_state
    return specs
