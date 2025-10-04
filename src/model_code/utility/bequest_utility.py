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
    sex,
    params,
):
    mu = jax.lax.select(education == 0, params["mu_low"], params["mu_high"])
    kappa = (
        params["kappa_low_men"] * (1 - education) * (1 - sex)
        + params["kappa_low_women"] * (1 - education) * sex
        + params["kappa_high_men"] * education * (1 - sex)
        + params["kappa_high_women"] * education * sex
    )
    unscaled_bequest_mu_not_one = ((wealth + kappa) ** (1 - mu) - 1) / (1 - mu)
    unscaled_bequest = jax.lax.select(
        jnp.allclose(mu, 1),
        jnp.log(wealth + kappa),
        unscaled_bequest_mu_not_one,
    )
    # bequest_scale = (
    #     params["bequest_scale_low_men"] * (1 - education) * (1 - sex)
    #     + params["bequest_scale_low_women"] * (1 - education) * sex
    #     + params["bequest_scale_high_men"] * education * (1 - sex)
    #     + params["bequest_scale_high_women"] * education * sex
    # )
    bequest_scale = params["bequest_scale"]
    return bequest_scale * unscaled_bequest


def marginal_utility_final_consume_all(wealth, education, sex, params):

    mu = jax.lax.select(education == 0, params["mu_low"], params["mu_high"])
    kappa = (
        params["kappa_low_men"] * (1 - education) * (1 - sex)
        + params["kappa_low_women"] * (1 - education) * sex
        + params["kappa_high_men"] * education * (1 - sex)
        + params["kappa_high_women"] * education * sex
    )
    # bequest_scale = (
    #     params["bequest_scale_low_men"] * (1 - education) * (1 - sex)
    #     + params["bequest_scale_low_women"] * (1 - education) * sex
    #     + params["bequest_scale_high_men"] * education * (1 - sex)
    #     + params["bequest_scale_high_women"] * education * sex
    # )
    bequest_scale = params["bequest_scale"]

    return bequest_scale * ((wealth + kappa) ** -mu)
