import jax.numpy as jnp


def health_transition(health, education, period, options):
    trans_mat = options["health_trans_mat"]
    prob_vector = trans_mat[education, period, health, :]
    return prob_vector
