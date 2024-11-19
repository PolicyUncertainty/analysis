import jax.numpy as jnp


def informed_transition(education, informed_state, options):
    """Transition function for informed state."""
    probability_informed = (
        options["informed_hazard_rate"][education] * (1 - informed_state)
        + informed_state
    )
    return jnp.array([1 - probability_informed, probability_informed])
