from jax import numpy as jnp
from model_code.wealth_and_budget.partner_income import calc_partner_income_after_ssc
from model_code.wealth_and_budget.pensions import calc_pensions_after_ssc
from model_code.wealth_and_budget.tax_and_ssc import calc_net_household_income
from model_code.wealth_and_budget.transfers import calc_child_benefits
from model_code.wealth_and_budget.transfers import calc_unemployment_benefits
from model_code.wealth_and_budget.wages import calc_labor_income_after_ssc


def budget_constraint(
    period,
    education,
    lagged_choice,  # d_{t-1}
    experience,
    partner_state,
    policy_state,  # current applicable SRA identifyer
    retirement_age_id,
    savings_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    options,
):
    # Recalculate experience
    max_exp_period = period + options["max_init_experience"]
    experience_years = max_exp_period * experience
    experience_years = jnp.minimum(experience_years, 44.0)

    # Calculate partner income
    partner_income_after_ssc = calc_partner_income_after_ssc(
        partner_state=partner_state, options=options, education=education, period=period
    )
    has_partner_int = (partner_state > 0).astype(int)

    # Income lagged choice 0
    unemployment_benefits = calc_unemployment_benefits(
        savings=savings_end_of_previous_period,
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        options=options,
    )

    # Income lagged choice 1
    labor_income_after_ssc = calc_labor_income_after_ssc(
        experience=experience_years,
        education=education,
        income_shock=income_shock_previous_period,
        options=options,
    )

    # Income from lagged choice 2
    retirement_income_after_ssc = calc_pensions_after_ssc(
        experience=experience_years,
        education=education,
        policy_state=policy_state,
        retirement_age_id=retirement_age_id,
        options=options,
    )

    # Select relevant income
    # bools of last period decision: income is payed in following period!
    was_worker = lagged_choice == 1
    was_retired = lagged_choice == 2

    # Aggregate over choice own income
    own_income_after_ssc = (
        was_worker * labor_income_after_ssc + was_retired * retirement_income_after_ssc
    )

    # Calculate total houshold net income
    total_net_income = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        options=options,
    )
    child_benefits = calc_child_benefits(
        education=education,
        has_partner_int=has_partner_int,
        period=period,
        options=options,
    )

    total_income = jnp.maximum(total_net_income + child_benefits, unemployment_benefits)
    # calculate beginning of period wealth M_t
    wealth = (
        1 + params["interest_rate"]
    ) * savings_end_of_previous_period + total_income

    return wealth
