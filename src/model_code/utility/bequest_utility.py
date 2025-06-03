import jax
import jax.numpy as jnp


def create_final_period_utility_functions():
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


def utility_final_consume_all(
    wealth,
    sex,
    params,
):
    wealth_shift = 1 + wealth
    mu = jax.lax.select(sex == 0, params["mu_men"], params["mu_women"])
    unscaled_bequest_mu_not_one = (wealth_shift ** (1 - mu) - 1) / (1 - mu)
    unscaled_bequest = jax.lax.select(
        jnp.allclose(mu, 1),
        jnp.log(wealth_shift),
        unscaled_bequest_mu_not_one,
    )

    bequest_scale = params["bequest_scale"]
    return bequest_scale * unscaled_bequest


def marginal_utility_final_consume_all(wealth, sex, params):
    wealth_shift = 1 + wealth

    mu = jax.lax.select(sex == 0, params["mu_men"], params["mu_women"])
    bequest_scale = params["bequest_scale"]
    return bequest_scale * (wealth_shift**-mu)
