import jax
import jax.numpy as jnp

from model_code.utility.bequest_utility import (
    marginal_utility_final_consume_all,
    utility_final_consume_all,
)


def create_utility_functions():
    return {
        "utility": utility_func,
        "marginal_utility": marginal_utility_func,
        "inverse_marginal_utility": inverse_marginal_func,
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
        education=education,
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
    mu = jax.lax.select(
        education == 1, on_true=params["mu_high"], on_false=params["mu_low"]
    )
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
    # We keep track in the model of the individual account. So if you have a partner and choose consumption, then you are
    # actually choosing double the consumption or vice versa consumption only costs half in your individual account if you
    # have a partner. A model trick to not have a transition equation dependent on lagged partner.
    has_partner_int = (partner_state > 0).astype(int)
    wealth_mult = 1 + has_partner_int
    # compute utility. The felicity is multiplied by cons_scale so that
    # per-equivalent consumption follows the standard Euler equation and total
    # consumption always tilts toward the larger household, independent of mu --
    # see docs/consumption_allocation.md.
    scaled_consumption = (wealth_mult * consumption) * eta / cons_scale
    utility_mu_not_one = cons_scale * (scaled_consumption ** (1 - mu) - 1) / (1 - mu)

    utility = jax.lax.select(
        jnp.allclose(mu, 1),
        cons_scale * jnp.log(scaled_consumption),
        utility_mu_not_one,
    )
    return utility


def marginal_utility_func(
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
    marginal_utility_alive = marginal_utility_function_alive(
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
    marginal_utility_death = marginal_utility_final_consume_all(
        wealth=consumption,
        education=education,
        params=params,
    )
    death_bool = health == model_specs["death_health_var"]
    marginal_utility = jax.lax.select(
        death_bool, marginal_utility_death, marginal_utility_alive
    )
    return marginal_utility


def marginal_utility_function_alive(
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
    mu = jax.lax.select(
        education == 1, on_true=params["mu_high"], on_false=params["mu_low"]
    )
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
    has_partner_int = (partner_state > 0).astype(int)
    wealth_mult = 1 + has_partner_int

    # Felicity is cons_scale * ((wealth_mult*c*eta/cons_scale)**(1-mu) - 1)/(1-mu);
    # its exact derivative wrt consumption is wealth_mult * eta *
    # (wealth_mult*c*eta/cons_scale)**(-mu). We drop the wealth_mult chain-rule
    # factor -- dcegm's Euler-equation solver hardcodes the marginal return on
    # savings as 1 + interest_rate and never differentiates budget_constraint,
    # so dropping it is what reproduces the transition-based (halve on divorce,
    # double on marriage) economics exactly. See the dcegm guide "Implementing a
    # divorce/marriage transition without a lagged partner state"
    # (docs/source/guides/) for the derivation and equivalence proof.
    marg_util_mu_not_one = (
        cons_scale
        * ((eta / cons_scale) ** (1 - mu))
        * ((wealth_mult * consumption) ** (-mu))
    )

    marg_util = jax.lax.select(
        jnp.allclose(mu, 1),
        cons_scale / (wealth_mult * consumption),
        marg_util_mu_not_one,
    )

    return marg_util


def inverse_marginal_func(
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
    mu = jax.lax.select(
        education == 1, on_true=params["mu_high"], on_false=params["mu_low"]
    )
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
    has_partner_int = (partner_state > 0).astype(int)
    wealth_mult = 1 + has_partner_int

    # Inverts marginal_utility_function_alive above:
    # m = cons_scale * (eta/cons_scale)**(1-mu) * (wealth_mult*c)**(-mu)
    # => c = cons_scale * m**(-1/mu) * eta**((1-mu)/mu) / wealth_mult
    consumption_mu_not_one = (
        cons_scale
        * marginal_utility ** (-1 / mu)
        * eta ** ((1 - mu) / mu)
        / wealth_mult
    )
    consumption = jax.lax.select(
        jnp.allclose(mu, 1),
        cons_scale / (wealth_mult * marginal_utility),
        consumption_mu_not_one,
    )
    return consumption


def disutility_work(
    period, choice, sex, education, partner_state, health, params, model_specs
):
    # choice booleans
    retired = choice == 0
    is_unemployed = choice == 1
    is_working_part_time = choice == 2
    is_working_full_time = choice == 3
    partner_retired = partner_state == 2

    good_health = health == model_specs["good_health_var"]
    bad_health = health == model_specs["bad_health_var"]
    disabled_health = health == model_specs["disabled_health_var"]

    # # Men's disutility parameters by health (no longer education-specific)
    # disutil_ft_work_men = (
    #     params["disutil_ft_work_bad_men"] * (1 - good_health)
    #     + params["disutil_ft_work_good_men"] * good_health
    # )
    #
    # disutil_unemployment_men = params[
    #     "disutil_unemployed_good_men"
    # ] * good_health + params["disutil_unemployed_bad_men"] * (1 - good_health)

    disutil_ft_work_men = (
        params["disutil_ft_work_high_bad_men"] * bad_health * education
        + params["disutil_ft_work_low_bad_men"] * bad_health * (1 - education)
        + params["disutil_ft_work_high_good_men"] * good_health * education
        + params["disutil_ft_work_low_good_men"] * good_health * (1 - education)
        + params["disutil_ft_work_disabled_men"] * disabled_health
    )

    disutil_unemployment_men = (
        params["disutil_unemployed_high_good_men"] * good_health * education
        + params["disutil_unemployed_low_good_men"] * good_health * (1 - education)
        + params["disutil_unemployed_high_bad_men"] * bad_health * education
        + params["disutil_unemployed_low_bad_men"] * bad_health * (1 - education)
        + params["disutil_unemployed_disabled_men"] * disabled_health
    )

    disutil_retirement_men = params["disutil_partner_retired_men"]

    exp_factor_men = (
        disutil_unemployment_men * is_unemployed
        + disutil_ft_work_men * is_working_full_time
        + partner_retired * disutil_retirement_men * retired
    )

    # Women's disutility parameters by health (no longer education-specific)
    disutil_ft_work_women = (
        params["disutil_ft_work_good_women"] * good_health
        + params["disutil_ft_work_bad_women"] * bad_health
        + params["disutil_ft_work_disabled_women"] * disabled_health
    )

    disutil_pt_work_women = (
        params["disutil_pt_work_good_women"] * good_health
        + params["disutil_pt_work_bad_women"] * bad_health
        + params["disutil_pt_work_disabled_women"] * disabled_health
    )

    # Children disutility remains education-specific as it's conceptually different
    disutil_children = params["disutil_children_ft_work_high"] * education + params[
        "disutil_children_ft_work_low"
    ] * (1 - education)

    has_partner_int = (partner_state > 0).astype(int)
    nb_children = model_specs["children_by_state"][
        sex, education, has_partner_int, period
    ]
    disutil_children_ft = disutil_children * nb_children

    disutil_unemployment_women = (
        params["disutil_unemployed_good_women"] * good_health
        + params["disutil_unemployed_bad_women"] * bad_health
        + params["disutil_unemployed_disabled_women"] * disabled_health
    )

    disutil_retirement_women = params["disutil_partner_retired_women"]

    exp_factor_women = (
        disutil_unemployment_women * is_unemployed
        + disutil_pt_work_women * is_working_part_time
        + (disutil_ft_work_women + disutil_children_ft) * is_working_full_time
        + partner_retired * disutil_retirement_women * retired
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
