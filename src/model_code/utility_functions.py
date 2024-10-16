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


def utility_func_old(
    consumption, partner_state, education, period, choice, params, options
):
    """Calculate the choice specific utility."""
    cons_scale = consumption_scale(partner_state, education, period, options)
    # Reading parameters
    mu = params["mu"]
    dis_util_work = (
        params["dis_util_work_low"] * (1 - education)
        + params["dis_util_work_high"] * education
    )
    dis_util_unemployed = (
        params["dis_util_unemployed_low"] * (1 - education)
        + params["dis_util_unemployed_high"] * education
    )
    # Check which choice we have
    is_working = choice == 1
    is_unemployed = choice == 0
    # Select which dis-utility to use. Retirement is 0 as baseline
    dis_utility = dis_util_work * is_working + dis_util_unemployed * is_unemployed
    # Compute utility
    utility = ((consumption / cons_scale) ** (1 - mu) - 1) / (1 - mu) - dis_utility
    return utility


def marg_utility_old(consumption, partner_state, education, period, params, options):
    cons_scale = consumption_scale(partner_state, education, period, options)

    mu = params["mu"]
    marg_util = (consumption / cons_scale) ** -mu * (1 / cons_scale)
    return marg_util


def inverse_marginal_old(
    marginal_utility, partner_state, education, period, params, options
):
    cons_scale = consumption_scale(partner_state, education, period, options)

    mu = params["mu"]
    return ((marginal_utility * cons_scale) ** (-1 / mu)) * cons_scale


def utility_final_consume_all(
    wealth,
    params,
):
    bequest_scale = params["bequest_scale"]
    return bequest_scale * (wealth ** (1 - params["mu"]) / (1 - params["mu"]))


def marginal_utility_final_consume_all(wealth, params):
    mu = params["mu"]
    bequest_scale = params["bequest_scale"]
    return bequest_scale * (wealth**-mu)


def consumption_scale(partner_state, education, period, options):
    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][0, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)


def utility_func(
    consumption, partner_state, education, period, choice, params, options
):
    """Calculate the choice specific cobb-douglas utility, i.e. u = ((c*eta/consumption_scale)^(1-mu))/(1-mu) .""" 
    # gather params
    mu = params["mu"]
    eta = disutility_work(choice, education, params)
    cons_scale = consumption_scale(partner_state, education, period, options)
    # compute utility
    utility = (consumption * eta / cons_scale) ** (1 - mu) / (1 - mu)
    return utility

def disutility_work (choice, education, params):
    # reading parameters
    dis_util_work = (
        params["dis_util_work_low"] * (1 - education)
        + params["dis_util_work_high"] * education
    )
    dis_util_unemployed = (
        params["dis_util_unemployed_low"] * (1 - education)
        + params["dis_util_unemployed_high"] * education
    )

    # dis_util_only_partner_retired = params["dis_util_only_partner_retired"]

    # choice booleans
    is_working = choice == 1
    is_unemployed = choice == 0
    # partner_retired = partner_state == 0

    # compute eta
    disutility = jnp.exp(-dis_util_work * is_working  - dis_util_unemployed* is_unemployed)
    # disutility = jnp.exp(-dis_util_work * is_working - dis_util_unemployed * is_unemployed - dis_util_only_partner_retired * partner_retired)
    return disutility

def marg_utility(consumption, partner_state, education, period, choice, params, options):
    cons_scale = consumption_scale(partner_state, education, period, options)
    mu = params["mu"]
    eta = disutility_work(choice, education, params)
    marg_util = ((eta/cons_scale) ** (1-mu)) * (consumption ** (-mu))
    return marg_util

def inverse_marginal(marginal_utility, partner_state, education, period, choice, params, options):
    cons_scale = consumption_scale(partner_state, education, period, options)
    mu = params["mu"]
    eta = disutility_work(choice, education, params)
    consumption = marginal_utility ** (-1/mu) * (eta/cons_scale) ** ((1-mu)/mu)
    return consumption
