import sys
from pathlib import Path

src_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(src_folder))
import pytest
from model_code.utility_functions import utility_func, inverse_marginal, marg_utility
import numpy as np
from itertools import product

MU_GRID = np.linspace(0.1, 0.9, 3)
CONSUMPTION_GRID = np.linspace(10, 100, 5)
DISUTIL_UNEMPLOYED_GRID = np.linspace(0.1, 0.9, 3)
DISUTIL_WORK_GRID = np.linspace(0.1, 0.9, 3)


@pytest.mark.parametrize(
    "consumption, dis_util_work, dis_util_unemployed, mu",
    list(
        product(CONSUMPTION_GRID, DISUTIL_WORK_GRID, DISUTIL_UNEMPLOYED_GRID, MU_GRID)
    ),
)
def test_utility_func(consumption, dis_util_work, dis_util_unemployed, mu):
    params = {
        "mu": mu,
        "dis_util_work": dis_util_work,
        "dis_util_unemployed": dis_util_unemployed,
    }
    cons_utility = consumption ** (1 - mu) / (1 - mu)

    np.testing.assert_almost_equal(
        utility_func(consumption=consumption, choice=1, params=params),
        cons_utility - dis_util_work,
    )
    np.testing.assert_almost_equal(
        utility_func(consumption=consumption, choice=0, params=params),
        cons_utility - dis_util_unemployed,
    )
