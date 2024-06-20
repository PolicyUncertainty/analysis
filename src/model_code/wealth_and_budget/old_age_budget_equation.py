import jax.numpy as jnp
from model_code.wealth_and_budget.tax_and_transfer import calc_net_income_pensions
from model_code.wealth_and_budget.tax_and_transfer import calc_unemployment_benefits


def old_age_budget_constraint(
    experience,
    deduction_state,
    savings_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    options,
):
    # fetch necessary parameters (gammas for wage, pension_point_value & ERP for pension)
    pension_point_value = options["pension_point_value"]
    ERP = options["early_retirement_penalty"]

    deduction = jnp.take(options["deduction_state_values"], deduction_state)

    # calculate applicable SRA and pension deduction/increase factor
    # (malus for early retirement, bonus for late retirement)
    # In the old age problem we have a deduction state, which corresponds to difference
    # between actual retirement age and SRA at resolution
    pension_factor = 1 - deduction * ERP
    retirement_income_gross = pension_point_value * experience * pension_factor * 12
    retirement_income = calc_net_income_pensions(retirement_income_gross, options)

    unemployment_benefits = calc_unemployment_benefits(
        savings_end_of_previous_period=savings_end_of_previous_period,
        options=options,
    )

    income = jnp.maximum(retirement_income, unemployment_benefits)

    # calculate beginning of period wealth M_t
    wealth = (1 + params["interest_rate"]) * savings_end_of_previous_period + income

    return wealth
