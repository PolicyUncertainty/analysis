import sys
from itertools import product

import numpy as np
import pytest

from model_code.wealth_and_budget.budget_equation import budget_constraint
from model_code.wealth_and_budget.wages import calc_net_income_working
from model_code.wealth_and_budget.pensions import calc_net_income_pensions
from model_code.derive_specs import generate_derived_and_data_derived_specs
from set_paths import create_path_dict

SAVINGS_GRID = np.linspace(10, 100, 5)
INTEREST_RATE_GRID = np.linspace(0.01, 0.1, 2)

BENEFITS_GRID = np.linspace(10, 100, 5)


@pytest.mark.parametrize(
    "unemployment_benefits, savings, interest_rate",
    list(product(BENEFITS_GRID, SAVINGS_GRID, INTEREST_RATE_GRID)),
)
def test_budget_unemployed(unemployment_benefits, savings, interest_rate):
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    specs["unemployment_benefits"] = unemployment_benefits

    params = {"interest_rate": interest_rate}
    wealth = budget_constraint(
        education=0,
        lagged_choice=0,
        experience=30,
        policy_state=0,
        retirement_age_id=0,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=specs,
    )
    if savings < specs["unemployment_wealth_thresh"]:
        np.testing.assert_almost_equal(
            wealth, savings * (1 + interest_rate) + unemployment_benefits * 12
        )
    else:
        np.testing.assert_almost_equal(wealth, savings * (1 + interest_rate))


GAMMA_GRID = np.linspace(0.1, 0.9, 3)
EXP_GRID = np.linspace(10, 30, 3, dtype=int)
EDUCATION_GRID = [0, 1]
INCOME_SHOCK_GRID = np.linspace(-0.5, 0.5, 3)

@pytest.mark.parametrize(
    "gamma, income_shock, experience, interest_rate, savings, education",
    list(
        product(
            GAMMA_GRID,
            INCOME_SHOCK_GRID,
            EXP_GRID,
            INTEREST_RATE_GRID,
            SAVINGS_GRID,
            EDUCATION_GRID,
        )
    ),
)
def test_budget_worker(
    gamma, income_shock, experience, interest_rate, savings, education
):
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

    gamma_array = np.array([gamma, gamma - 0.01])
    specs["gamma_0"] = gamma_array
    specs["gamma_1"] = gamma_array

    params = {"interest_rate": interest_rate}
    wealth = budget_constraint(
        education=education,
        lagged_choice=1,
        experience=experience,
        policy_state=0,
        retirement_age_id=0,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=income_shock,
        params=params,
        options=specs,
    )
    labor_income = (
        np.exp(
            gamma_array[education]
            + gamma_array[education] * np.log(experience + 1)
            + income_shock
        )
        / specs["wealth_unit"]
    )
    if labor_income < specs["min_wage"]:
        labor_income = specs["min_wage"]
    net_labor_income = calc_net_income_working(labor_income * 12, specs)
    np.testing.assert_almost_equal(
        wealth, savings * (1 + interest_rate) + net_labor_income
    )


EXP_GRID = np.linspace(10, 30, 3, dtype=int)
POLICY_STATE_GRID = np.linspace(0, 2, 3, dtype=int)
RET_AGE_GRID = np.linspace(0, 2, 3, dtype=int)


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
    education = 0
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

    actual_retirement_age = specs["min_ret_age"] + ret_age_id

    params = {"interest_rate": interest_rate}
    wealth = budget_constraint(
        education=education,
        lagged_choice=2,
        experience=exp,
        policy_state=policy_state,
        retirement_age_id=ret_age_id,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=specs,
    )
    pension_point_value = specs["pension_point_value_by_edu_exp"][education, exp]
    SRA_at_resolution = specs["min_SRA"] + policy_state * specs["SRA_grid_size"]
    pension_factor = (
        1
        - (actual_retirement_age - SRA_at_resolution)
        * specs["early_retirement_penalty"]
    )
    retirement_income = calc_net_income_pensions(
        pension_point_value * pension_factor * exp * 12, specs
    )
    if savings < specs["unemployment_wealth_thresh"]:
        retirement_income = np.maximum(
            retirement_income, specs["unemployment_benefits"] * 12
        )

    np.testing.assert_almost_equal(
        wealth, savings * (1 + interest_rate) + retirement_income
    )
