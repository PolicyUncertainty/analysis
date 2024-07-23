import numpy as np
from jax import numpy as jnp
from wealth_and_budget.pensions import calc_pensions
from wealth_and_budget.tax_and_transfers import calc_unemployment_benefits
from wealth_and_budget.wages import calc_labor_income


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


def create_savings_grid():
    """Create a saving grid with sections."""
    section_1 = np.arange(start=0, stop=10, step=0.5)  # 20
    section_2 = np.arange(start=10, stop=50, step=1)  # 40
    section_3 = np.arange(start=50, stop=100, step=5)  # 10
    section_4 = np.arange(start=100, stop=500, step=20)  # 20
    section_5 = np.arange(start=500, stop=1000, step=100)  # 5
    savings_grid = np.concatenate(
        [section_1, section_2, section_3, section_4, section_5]
    )
    return savings_grid
