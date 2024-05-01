import jax.numpy as jnp
from model_code.wealth_and_budget.tax_and_transfer import calc_net_income_pensions
from model_code.wealth_and_budget.tax_and_transfer import calc_net_income_working
from model_code.wealth_and_budget.tax_and_transfer import calc_unemployment_benefits


def budget_constraint(
    lagged_choice,  # d_{t-1}
    experience,
    policy_state,  # current applicable SRA identifyer
    retirement_age_id,
    savings_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    options,
):
    # fetch necessary parameters (gammas for wage, pension_point_value & ERP for pension)
    gamma_0 = options["gamma_0"]
    gamma_1 = options["gamma_1"]
    gamma_2 = options["gamma_2"]
    pension_point_value = options["pension_point_value"]
    ERP = options["early_retirement_penalty"]

    # Calculate deduction for pension
    deduction = calc_deduction(policy_state, retirement_age_id, options)

    # calculate applicable SRA and pension deduction/increase factor
    # (malus for early retirement, bonus for late retirement)
    pension_factor = 1 - deduction * ERP
    retirement_income_gross = pension_point_value * experience * pension_factor * 12
    retirement_income = calc_net_income_pensions(retirement_income_gross)

    unemployment_benefits = calc_unemployment_benefits(
        savings_end_of_previous_period=savings_end_of_previous_period,
        options=options,
    )

    # Labor income
    labor_income = (
        gamma_0
        + gamma_1 * experience
        + gamma_2 * experience**2
        + income_shock_previous_period
    )
    labor_income_with_min = jnp.maximum(labor_income, options["min_wage"]) * 12
    net_labor_income = calc_net_income_working(labor_income_with_min)

    # bools of last period decision: income is payed in following period!
    was_worker = lagged_choice == 1
    was_retired = lagged_choice == 2

    income = jnp.maximum(
        was_worker * net_labor_income + was_retired * retirement_income,
        unemployment_benefits,
    )

    # calculate beginning of period wealth M_t
    wealth = (1 + params["interest_rate"]) * savings_end_of_previous_period + income

    return wealth


def calc_deduction(policy_state, retirement_age_id, options):
    SRA_at_resolution = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    actual_retirement_age = options["min_ret_age"] + retirement_age_id
    return actual_retirement_age - SRA_at_resolution
