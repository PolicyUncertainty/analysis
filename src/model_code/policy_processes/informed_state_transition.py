import jax
import jax.numpy as jnp


def informed_transition(choice, education, informed, model_specs):
    """Transition function for informed state. We use this function in the simulation
    and not in the solution of the model, as all individuals don't account for a
    possible change in their information set.

    When they retire, however we let them transfer to the informed state with a certain
    probability.

    """
    # # If an individual is already informed, they stay informed
    # # If they are uninformed, there is an education specific hazard rate
    # # to become informed
    # uninformed = informed == 0
    # hazard_rate_edu = model_specs["informed_hazard_rate"][education]
    # prob_informed = jax.lax.select(uninformed, hazard_rate_edu, 1.0)
    # prob_vector = jnp.array([1 - prob_informed, prob_informed])
    #
    # # If retirement is chosen the transition vector is ([0, 1]) as they transfer with
    # # certainty to the informed state
    # retirement_bool = choice == 0
    # certain_transition = jnp.array([0, 1], dtype=prob_vector.dtype)
    # prob_vector = jax.lax.select(retirement_bool, certain_transition, prob_vector)

    return jnp.array(
        [0, 1], dtype=jnp.float64
    )  # Always transition to informed state on retirement
