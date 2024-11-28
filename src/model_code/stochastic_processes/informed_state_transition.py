import jax
import jax.numpy as jnp


def informed_transition(choice, education, informed, options):
    """Transition function for informed state. We use this function in the simulation
    and not in the solution of the model, as all individuals don't account for a
    possible change in their information set.

    When they retire, however we let them transfer to the informed state with a certain
    probability.

    """
    # If an individual is already informed, they stay informed
    # If they are uninformed, there is an education specific hazard rate
    # to become informed
    uninformed = informed == 0
    hazard_rate_edu = options["informed_hazard_rate"][education]
    prob_informed = jax.lax.select(uninformed, hazard_rate_edu, 1.0)
    prob_vector = jnp.array([1 - prob_informed, prob_informed])

    # If retirement is choosen the transition vector is ([0, 1]) as they transfer with
    # certainty to the informed state
    retirement_bool = choice == 0
    certaint_transition = jnp.array([0, 1], dtype=prob_vector.dtype)
    prob_vector = jax.lax.select(retirement_bool, certaint_transition, prob_vector)

    return prob_vector
