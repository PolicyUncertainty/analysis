import jax.numpy as jnp


def health_transition(health, education, period, options):
    age = period + options["start_age"]
    trans_mat = options["health_trans_mat"]
    prob_good_health = trans_mat[education, age, health, 1]
    return jnp.array([1 - prob_good_health, prob_good_health])
