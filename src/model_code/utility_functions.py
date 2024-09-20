import jax.numpy as jnp


def create_utility_functions():
    return {
        "utility": utility_func,
        "inverse_marginal_utility": inverse_marginal,
        "marginal_utility": marg_utility,
    }


def create_final_period_utility_functions():
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


def utility_func(
    consumption, partner_state, education, period, choice, params, options
):
    """Calculate the choice specific utility."""
    cons_scale = consumption_scale(partner_state, education, period, options)
    # Reading parameters
    mu = params["mu"]
    dis_util_work = params["dis_util_work"][education]
    dis_util_unemployed = params["dis_util_unemployed"][education]
    # Check which choice we have
    is_working = choice == 1
    is_unemployed = choice == 0
    # Select which dis-utility to use. Retirement is 0 as baseline
    dis_utility = dis_util_work * is_working + dis_util_unemployed * is_unemployed
    # Compute utility
    utility = ((consumption / cons_scale) ** (1 - mu) - 1) / (1 - mu) - dis_utility
    return utility


def marg_utility(consumption, partner_state, education, period, params, options):
    cons_scale = consumption_scale(partner_state, education, period, options)

    mu = params["mu"]
    marg_util = (consumption / cons_scale) ** -mu * (1 / cons_scale)
    return marg_util


def inverse_marginal(
    marginal_utility, partner_state, education, period, params, options
):
    cons_scale = consumption_scale(partner_state, education, period, options)

    mu = params["mu"]
    return ((marginal_utility * cons_scale) ** (-1 / mu)) * cons_scale


def utility_final_consume_all(
    choice,
    resources,
    params,
    options,
):
    mu = params["mu"]
    bequest_scale = params["bequest_scale"]
    return bequest_scale * (resources ** (1 - mu) / (1 - mu))


def marginal_utility_final_consume_all(choice, resources, params, options):
    mu = params["mu"]
    bequest_scale = params["bequest_scale"]
    return bequest_scale * (resources**-mu)


def consumption_scale(partner_state, education, period, options):
    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][0, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
