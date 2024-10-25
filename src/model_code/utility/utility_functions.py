import jax.numpy as jnp


def create_utility_functions():
    return {
        "utility": utility_func,
        "inverse_marginal_utility": inverse_marginal,
        "marginal_utility": marg_utility,
    }


def utility_func(
    consumption, partner_state, education, period, choice, params, options
):
    """Calculate the choice specific cobb-douglas utility, i.e. u =
    ((c*eta/consumption_scale)^(1-mu))/(1-mu) ."""
    # gather params
    mu = params["mu"]
    eta = disutility_work(choice, education, params)
    cons_scale = consumption_scale(partner_state, education, period, options)
    # compute utility
    utility = (consumption * eta / cons_scale) ** (1 - mu) / (1 - mu)
    return utility


def marg_utility(
    consumption, partner_state, education, period, choice, params, options
):
    cons_scale = consumption_scale(partner_state, education, period, options)
    mu = params["mu"]
    eta = disutility_work(choice, education, params)
    marg_util = ((eta / cons_scale) ** (1 - mu)) * (consumption ** (-mu))
    return marg_util


def inverse_marginal(
    marginal_utility, partner_state, education, period, choice, params, options
):
    cons_scale = consumption_scale(partner_state, education, period, options)
    mu = params["mu"]
    eta = disutility_work(choice, education, params)
    consumption = marginal_utility ** (-1 / mu) * (eta / cons_scale) ** ((1 - mu) / mu)
    return consumption


def disutility_work(choice, education, params):
    # reading parameters
    dis_util_ft_work = (
        params["dis_util_ft_work_low"] * (1 - education)
        + params["dis_util_ft_work_high"] * education
    )
    dis_util_pt_work = (
        params["dis_util_pt_work_low"] * (1 - education)
        + params["dis_util_pt_work_high"] * education
    )

    dis_util_unemployed = (
        params["dis_util_unemployed_low"] * (1 - education)
        + params["dis_util_unemployed_high"] * education
    )

    # dis_util_only_partner_retired = params["dis_util_only_partner_retired"]

    # choice booleans
    is_unemployed = choice == 1
    is_working_part_time = choice == 2
    is_working_full_time = choice == 3
    # partner_retired = partner_state == 0

    exp_factor = (
        dis_util_unemployed * is_unemployed
        + dis_util_pt_work * is_working_part_time
        + dis_util_ft_work * is_working_full_time
    )

    # compute eta
    disutility = jnp.exp(-exp_factor)
    # disutility = jnp.exp(-dis_util_work * is_working - dis_util_unemployed * is_unemployed - dis_util_only_partner_retired * partner_retired)
    return disutility


def consumption_scale(partner_state, education, period, options):
    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][0, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
