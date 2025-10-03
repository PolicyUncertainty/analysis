import jax.numpy as jnp


def logit_formula(x):
    return 1 / (1 + jnp.exp(-x))


def inv_logit_formula(p):
    return jnp.log(p / (1 - p))
