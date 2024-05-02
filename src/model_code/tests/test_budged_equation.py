import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

src_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(src_folder))

from wealth_and_budget.main_budget_equation import main_budget_constraint
from wealth_and_budget.tax_and_transfer import (
    calc_net_income_pensions,
    calc_net_income_working,
)

SAVINGS_GRID = np.linspace(10, 100, 5)
INTEREST_RATE_GRID = np.linspace(0.01, 0.1, 2)

BENEFITS_GRID = np.linspace(10, 100, 5)


@pytest.mark.parametrize(
    "unemployment_benefits, savings, interest_rate",
    list(product(BENEFITS_GRID, SAVINGS_GRID, INTEREST_RATE_GRID)),
)
def test_budget_unemployed(unemployment_benefits, savings, interest_rate):
    options = {
        "gamma_0": 1,
        "gamma_1": 1,
        "gamma_2": 1,
        "pension_point_value": 1,
        "early_retirement_penalty": 0.01,
        "min_SRA": 63,
        "SRA_grid_size": 0.5,
        "min_ret_age": 65,
        "unemployment_benefits": unemployment_benefits,
        "min_wage": 1.1,
        "unemployment_wealth_thresh": 25,
    }
    params = {"interest_rate": interest_rate}
    wealth = main_budget_constraint(
        lagged_choice=0,
        experience=30,
        policy_state=0,
        retirement_age_id=0,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=options,
    )
    if savings < options["unemployment_wealth_thresh"]:
        np.testing.assert_almost_equal(
            wealth, savings * (1 + interest_rate) + unemployment_benefits * 12
        )
    else:
        np.testing.assert_almost_equal(wealth, savings * (1 + interest_rate))


GAMMA_GRID = np.linspace(0.1, 0.9, 3)
EXP_GRID = np.linspace(10, 30, 3)


@pytest.mark.parametrize(
    "gamma, income_shock, experience, interest_rate, savings",
    list(
        product(GAMMA_GRID, BENEFITS_GRID, EXP_GRID, INTEREST_RATE_GRID, SAVINGS_GRID)
    ),
)
def test_budget_worker(gamma, income_shock, experience, interest_rate, savings):
    options = {
        "gamma_0": gamma,
        "gamma_1": gamma,
        "gamma_2": gamma,
        "pension_point_value": 1,
        "early_retirement_penalty": 0.01,
        "min_SRA": 63,
        "SRA_grid_size": 0.5,
        "min_ret_age": 65,
        "unemployment_benefits": 0,
        "min_wage": 100,
        "unemployment_wealth_thresh": 10,
    }
    params = {"interest_rate": interest_rate}
    wealth = main_budget_constraint(
        lagged_choice=1,
        experience=experience,
        policy_state=0,
        retirement_age_id=0,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=income_shock,
        params=params,
        options=options,
    )
    labor_income = gamma + gamma * experience + gamma * experience**2 + income_shock
    if labor_income < options["min_wage"]:
        labor_income = options["min_wage"]
    net_labor_income = calc_net_income_working(labor_income * 12)
    np.testing.assert_almost_equal(
        wealth, savings * (1 + interest_rate) + net_labor_income
    )


EXP_GRID = np.linspace(10, 30, 3)
POLICY_STATE_GRID = np.linspace(0, 2, 3)
RET_AGE_GRID = np.linspace(0, 2, 3)


@pytest.mark.parametrize(
    "interest_rate, savings, exp, policy_state, ret_age_id",
    list(
        product(
            INTEREST_RATE_GRID, SAVINGS_GRID, EXP_GRID, POLICY_STATE_GRID, RET_AGE_GRID
        )
    ),
)
def test_retiree(
    interest_rate,
    savings,
    exp,
    policy_state,
    ret_age_id,
):
    min_ret_age = 63
    min_SRA = 65
    SRA_grid_size = 0.25
    erp = 0.36
    point_value = 0.9
    actual_retirement_age = min_ret_age + ret_age_id
    options = {
        "gamma_0": 1,
        "gamma_1": 1,
        "gamma_2": 1,
        "pension_point_value": point_value,
        "early_retirement_penalty": erp,
        "min_SRA": min_SRA,
        "SRA_grid_size": SRA_grid_size,
        "min_ret_age": min_ret_age,
        "unemployment_benefits": 50,
        "min_wage": 100,
        "unemployment_wealth_thresh": 100,
    }
    params = {"interest_rate": interest_rate}
    wealth = main_budget_constraint(
        lagged_choice=2,
        experience=exp,
        policy_state=policy_state,
        retirement_age_id=ret_age_id,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=options,
    )
    pension_point_value = options["pension_point_value"]
    SRA_at_resolution = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    pension_factor = (
        1
        - (actual_retirement_age - SRA_at_resolution)
        * options["early_retirement_penalty"]
    )
    retirement_income = calc_net_income_pensions(
        pension_point_value * pension_factor * exp * 12
    )
    if savings < options["unemployment_wealth_thresh"]:
        retirement_income = np.maximum(
            retirement_income, options["unemployment_benefits"] * 12
        )

    np.testing.assert_almost_equal(
        wealth, savings * (1 + interest_rate) + retirement_income
    )
