import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

src_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(src_folder))

from model_code.budget_equation import budget_constraint

BENEFITS_GRID = np.linspace(10, 100, 5)
SAVINGS_GRID = np.linspace(10, 100, 5)
INTEREST_RATE_GRID = np.linspace(0.01, 0.1, 2)


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
    }
    params = {"interest_rate": interest_rate}
    wealth = budget_constraint(
        lagged_choice=0,
        experience=30,
        policy_state=0,
        retirement_age_id=0,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=options,
    )
    np.testing.assert_almost_equal(
        wealth, savings * (1 + interest_rate) + unemployment_benefits
    )
