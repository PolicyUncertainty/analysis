from jax import numpy as jnp
from model_code.wealth_and_budget.pensions import calc_pensions
from model_code.wealth_and_budget.tax_and_transfers import calc_unemployment_benefits
from model_code.wealth_and_budget.wages import calc_labor_income


def budget_constraint(
    education,
    lagged_choice,  # d_{t-1}
    experience,
    policy_state,  # current applicable SRA identifyer
    retirement_age_id,
    savings_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    options,
):
    # Income lagged choice 0
    unemployment_benefits = calc_unemployment_benefits(
        savings=savings_end_of_previous_period, options=options
    )

    # Income lagged choice 1
    labor_income = calc_labor_income(
        experience=experience,
        education=education,
        income_shock=income_shock_previous_period,
        options=options,
    )

    # Income from lagged choice 2
    retirement_income = calc_pensions(
        experience=experience,
        education=education,
        policy_state=policy_state,
        retirement_age_id=retirement_age_id,
        options=options,
    )

    # Select relevant income
    # bools of last period decision: income is payed in following period!
    was_worker = lagged_choice == 1
    was_retired = lagged_choice == 2

    # If individual was unemployed (lagged_choice == 0) or if income of according
    # choice is lower, then unemployment income is payed
    income = jnp.maximum(
        was_worker * labor_income + was_retired * retirement_income,
        unemployment_benefits,
    )

    # calculate beginning of period wealth M_t
    wealth = (1 + params["interest_rate"]) * savings_end_of_previous_period + income

    return wealth
