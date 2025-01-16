import jax
import jax.numpy as jnp
from model_code.utility.bequest_utility import utility_final_consume_all


def create_utility_functions():
    return {
        "utility": utility_func,
        "inverse_marginal_utility": inverse_marginal,
        "marginal_utility": marg_utility,
    }


def create_utility_functions_sim():
    return {
        "utility": utility_func_sim,
    }


def utility_func_sim(
    consumption, sex, partner_state, education, health, period, choice, params, options
):
    utility_alive = utility_func(
        consumption=consumption,
        sex=sex,
        partner_state=partner_state,
        education=education,
        health=health,
        period=period,
        choice=choice,
        params=params,
        options=options,
    )
    utility_death = utility_final_consume_all(
        wealth=consumption,
        params=params,
    )
    death_bool = health == 2
    utility = jax.lax.select(death_bool, utility_death, utility_alive)
    return utility


def utility_func(
    consumption, sex, partner_state, education, health, period, choice, params, options
):
    """Calculate the choice specific cobb-douglas utility, i.e. u =
    ((c*eta/consumption_scale)^(1-mu))/(1-mu) ."""
    # gather params
    mu = params["mu"]
    eta = disutility_work(
        period=period,
        choice=choice,
        sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    cons_scale = consumption_scale(
        partner_state=partner_state,
        sex=sex,
        education=education,
        period=period,
        options=options,
    )
    # compute utility
    utility = (consumption * eta / cons_scale) ** (1 - mu) / (1 - mu)
    return utility


def marg_utility(
    consumption, partner_state, sex, education, health, period, choice, params, options
):
    cons_scale = consumption_scale(
        partner_state=partner_state,
        sex=sex,
        education=education,
        period=period,
        options=options,
    )
    mu = params["mu"]
    eta = disutility_work(
        period=period,
        choice=choice,
        sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    marg_util = ((eta / cons_scale) ** (1 - mu)) * (consumption ** (-mu))
    return marg_util


def inverse_marginal(
    marginal_utility,
    partner_state,
    education,
    sex,
    health,
    period,
    choice,
    params,
    options,
):
    cons_scale = consumption_scale(
        partner_state=partner_state,
        sex=sex,
        education=education,
        period=period,
        options=options,
    )
    mu = params["mu"]
    eta = disutility_work(
        period=period,
        choice=choice,
        sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        options=options,
    )
    consumption = marginal_utility ** (-1 / mu) * (eta / cons_scale) ** ((1 - mu) / mu)
    return consumption


def disutility_work(
    period, choice, sex, education, partner_state, health, params, options
):
    # choice booleans
    is_unemployed = choice == 1
    is_working_part_time = choice == 2
    is_working_full_time = choice == 3
    # partner_retired = partner_state == 0

    # reading parameters
    disutil_ft_work_men = (
        params["disutil_ft_work_bad_men"] * (1 - health)
        + params["disutil_ft_work_good_men"] * health
    )
    exp_factor_men = (
        params["disutil_unemployed_men"] * is_unemployed
        # + disutil_pt_work * is_working_part_time
        + disutil_ft_work_men * is_working_full_time
        # + partner_retired * disutil_only_partner_retired
    )

    disutil_ft_work_women = (
        params["disutil_ft_work_bad_women"] * (1 - health)
        + params["disutil_ft_work_good_women"] * health
    )
    disutil_pt_work_women = (
        params["disutil_pt_work_bad_women"] * (1 - health)
        + params["disutil_pt_work_good_women"] * health
    )
    has_partner_int = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][sex, education, has_partner_int, period]

    disutil_children_high = params["disutil_children_ft_work_high"] * nb_children
    disutil_children_low = params["disutil_children_ft_work_low"] * nb_children
    disutil_children_ft = disutil_children_high * education + disutil_children_low * (
        1 - education
    )

    exp_factor_women = (
        params["disutil_unemployed_women"] * is_unemployed
        + disutil_pt_work_women * is_working_part_time
        + (disutil_ft_work_women + disutil_children_ft) * is_working_full_time
    )

    # Select exponential factor by sex
    exp_factor = jax.lax.select(sex == 0, exp_factor_men, exp_factor_women)
    # compute eta
    disutility = jnp.exp(-exp_factor)
    return disutility


def consumption_scale(partner_state, sex, education, period, options):
    has_partner = (partner_state > 0).astype(int)
    nb_children = options["children_by_state"][sex, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
