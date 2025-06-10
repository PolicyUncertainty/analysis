import jax
import jax.numpy as jnp

from model_code.utility.bequest_utility import utility_final_consume_all


def create_utility_functions():
    return {
        "utility": utility_func,
        "inverse_marginal_utility": inverse_marginal,
        "marginal_utility": marg_utility,
    }


def utility_func(
    consumption,
    sex,
    partner_state,
    education,
    health,
    period,
    choice,
    params,
    model_specs,
):
    utility_alive = utility_func_alive(
        consumption=consumption,
        sex=sex,
        partner_state=partner_state,
        education=education,
        health=health,
        period=period,
        choice=choice,
        params=params,
        model_specs=model_specs,
    )
    utility_death = utility_final_consume_all(
        wealth=consumption,
        sex=sex,
        params=params,
    )
    death_bool = health == model_specs["death_health_var"]
    utility = jax.lax.select(death_bool, utility_death, utility_alive)
    return utility


def utility_func_alive(
    consumption,
    sex,
    partner_state,
    education,
    health,
    period,
    choice,
    params,
    model_specs,
):
    """Calculate the choice specific cobb-douglas utility, i.e. u =
    ((c*eta/consumption_scale)^(1-mu))/(1-mu) ."""
    # gather params
    mu = jax.lax.select(sex == 0, params["mu_men"], params["mu_women"])
    eta = disutility_work(
        period=period,
        choice=choice,
        sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        model_specs=model_specs,
    )
    cons_scale = consumption_scale(
        partner_state=partner_state,
        sex=sex,
        education=education,
        period=period,
        model_specs=model_specs,
    )
    # compute utility
    scaled_consumption = consumption * eta / cons_scale
    utility_mu_not_one = (scaled_consumption ** (1 - mu) - 1) / (1 - mu)

    utility = jax.lax.select(
        jnp.allclose(mu, 1),
        jnp.log(consumption * eta / cons_scale),
        utility_mu_not_one,
    )
    return utility


def marg_utility(
    consumption,
    partner_state,
    sex,
    education,
    health,
    period,
    choice,
    params,
    model_specs,
):
    cons_scale = consumption_scale(
        partner_state=partner_state,
        sex=sex,
        education=education,
        period=period,
        model_specs=model_specs,
    )
    mu = jax.lax.select(sex == 0, params["mu_men"], params["mu_women"])
    eta = disutility_work(
        period=period,
        choice=choice,
        sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        model_specs=model_specs,
    )
    marg_util_mu_not_one = ((eta / cons_scale) ** (1 - mu)) * (consumption ** (-mu))

    marg_util = jax.lax.select(
        jnp.allclose(mu, 1),
        1 / consumption,
        marg_util_mu_not_one,
    )

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
    model_specs,
):
    cons_scale = consumption_scale(
        partner_state=partner_state,
        sex=sex,
        education=education,
        period=period,
        model_specs=model_specs,
    )
    mu = jax.lax.select(sex == 0, params["mu_men"], params["mu_women"])
    eta = disutility_work(
        period=period,
        choice=choice,
        sex=sex,
        education=education,
        partner_state=partner_state,
        health=health,
        params=params,
        model_specs=model_specs,
    )
    consumption_mu_not_one = marginal_utility ** (-1 / mu) * (eta / cons_scale) ** (
        (1 - mu) / mu
    )
    consumption = jax.lax.select(
        jnp.allclose(mu, 1), 1 / marginal_utility, consumption_mu_not_one
    )
    return consumption


def disutility_work(
    period, choice, sex, education, partner_state, health, params, model_specs
):
    # choice booleans
    is_unemployed = choice == 1
    is_working_part_time = choice == 2
    is_working_full_time = choice == 3
    # partner_retired = partner_state == 0

    good_health = health == model_specs["good_health_var"]

    # reading parameters
    disutil_ft_work_men = (
        params["disutil_ft_work_bad_men"] * (1 - good_health)
        + params["disutil_ft_work_good_men"] * good_health
    )
    disutil_unemployment_men = params[
        "disutil_unemployed_good_men"
    ] * good_health + params["disutil_unemployed_bad_men"] * (1 - good_health)

    exp_factor_men = (
        disutil_unemployment_men * is_unemployed
        # + disutil_pt_work * is_working_part_time
        + disutil_ft_work_men * is_working_full_time
        # + partner_retired * disutil_only_partner_retired
    )

    disutil_ft_work_women = (
        params["disutil_ft_work_high_bad_women"] * (1 - good_health) * education
        + params["disutil_ft_work_low_bad_women"] * (1 - good_health) * (1 - education)
        + params["disutil_ft_work_high_good_women"] * good_health * education
        + params["disutil_ft_work_low_good_women"] * good_health * (1 - education)
    )
    disutil_pt_work_women = (
        params["disutil_pt_work_high_bad_women"] * (1 - good_health) * education
        + params["disutil_pt_work_low_bad_women"] * (1 - good_health) * (1 - education)
        + params["disutil_pt_work_high_good_women"] * good_health * education
        + params["disutil_pt_work_low_good_women"] * good_health * (1 - education)
    )

    disutil_children = params["disutil_children_ft_work_high"] * education + params[
        "disutil_children_ft_work_low"
    ] * (1 - education)

    has_partner_int = (partner_state > 0).astype(int)
    nb_children = model_specs["children_by_state"][
        sex, education, has_partner_int, period
    ]
    disutil_children_ft = disutil_children * nb_children

    disutil_unemployment = params["disutil_unemployed_high_women"] * education + params[
        "disutil_unemployed_low_women"
    ] * (1 - education)

    exp_factor_women = (
        disutil_unemployment * is_unemployed
        + disutil_pt_work_women * is_working_part_time
        + (disutil_ft_work_women + disutil_children_ft) * is_working_full_time
    )

    # Select exponential factor by sex
    exp_factor = jax.lax.select(sex == 0, exp_factor_men, exp_factor_women)
    # compute eta
    disutility = jnp.exp(-exp_factor)
    return disutility


def consumption_scale(partner_state, sex, education, period, model_specs):
    has_partner = (partner_state > 0).astype(int)
    nb_children = model_specs["children_by_state"][sex, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    return jnp.sqrt(hh_size)
