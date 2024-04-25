import jax.numpy as jnp
import numpy as np

from tax_and_transfer import calc_net_income_pensions, calc_unemployment_benefits

def budget_constraint(
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

    # calculate applicable SRA and pension deduction/increase factor
    # (malus for early retirement, bonus for late retirement)
    # In the old age problem we have a deduction state, which corresponds to difference
    # between actual retirement age and SRA at resolution
    pension_factor = 1 - deduction_state * ERP
    retirement_income_gross = pension_point_value * experience * pension_factor * 12
    retirement_income = calc_net_income_pensions(retirement_income_gross)

    unemployment_benefits = calc_unemployment_benefits(
        savings_end_of_previous_period=savings_end_of_previous_period,
        options=options,
    )

    income = jnp.maximum(retirement_income,unemployment_benefits)

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