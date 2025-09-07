import jax
import jax.numpy as jnp


def create_final_period_utility_functions():
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


def utility_final_consume_all(
    wealth,
    education,
    params,
):
    mu = jax.lax.select(education == 0, params["mu_low"], params["mu_high"])
    unscaled_bequest_mu_not_one = (wealth ** (1 - mu) - 1) / (1 - mu)
    unscaled_bequest = jax.lax.select(
        jnp.allclose(mu, 1),
        jnp.log(wealth),
        unscaled_bequest_mu_not_one,
    )

    bequest_scale = params["bequest_scale"]
    return bequest_scale * unscaled_bequest


def marginal_utility_final_consume_all(wealth, education, params):

    mu = jax.lax.select(education == 0, params["mu_low"], params["mu_high"])
    bequest_scale = params["bequest_scale"]
    return bequest_scale * (wealth**-mu)
