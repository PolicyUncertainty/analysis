import sys
from pathlib import Path

src_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(src_folder))
import pytest
from model_code.utility_functions import (
    utility_func,
    inverse_marginal,
    marg_utility,
    utility_final_consume_all,
    marginal_utility_final_consume_all,
)
import numpy as np
from itertools import product
import jax

jax.config.update("jax_enable_x64", True)

MU_GRID = np.linspace(0.1, 0.9, 3)
CONSUMPTION_GRID = np.linspace(10, 100, 5)
DISUTIL_UNEMPLOYED_GRID = np.linspace(0.1, 0.9, 3)
DISUTIL_WORK_GRID = np.linspace(0.1, 0.9, 3)
BEQUEST_SCALE = np.linspace(1, 4, 3)


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
    cons_utility = consumption ** (1 - mu) / (1 - mu) - 1 / (1 - mu)

    np.testing.assert_almost_equal(
        utility_func(consumption=consumption, choice=1, params=params),
        cons_utility - dis_util_work,
    )
    np.testing.assert_almost_equal(
        utility_func(consumption=consumption, choice=0, params=params),
        cons_utility - dis_util_unemployed,
    )


@pytest.mark.parametrize(
    "consumption, dis_util_work, dis_util_unemployed, mu",
    list(
        product(CONSUMPTION_GRID, DISUTIL_WORK_GRID, DISUTIL_UNEMPLOYED_GRID, MU_GRID)
    ),
)
def test_marginal_utility(consumption, dis_util_work, dis_util_unemployed, mu):
    params = {
        "mu": mu,
        "dis_util_work": dis_util_work,
        "dis_util_unemployed": dis_util_unemployed,
    }
    random_choice = np.random.choice(np.array([0, 1, 2]))
    marg_util_jax = jax.jacfwd(utility_func, argnums=0)(
        consumption, random_choice, params
    )
    marg_util_model = marg_utility(consumption, params)
    np.testing.assert_almost_equal(marg_util_jax, marg_util_model)


@pytest.mark.parametrize(
    "consumption, mu",
    list(product(CONSUMPTION_GRID, MU_GRID)),
)
def test_inv_marginal_utility(consumption, mu):
    params = {
        "mu": mu,
    }
    marg_util = marg_utility(consumption, params)
    np.testing.assert_almost_equal(inverse_marginal(marg_util, params), consumption)


@pytest.mark.parametrize(
    "consumption, mu, bequest_scale",
    list(product(CONSUMPTION_GRID, MU_GRID, BEQUEST_SCALE)),
)
def test_bequest(consumption, mu, bequest_scale):
    params = {
        "mu": mu,
        "bequest_scale": bequest_scale,
    }
    bequest = bequest_scale * (consumption ** (1 - mu) / (1 - mu))
    random_choice = np.random.choice(np.array([0, 1, 2]))
    np.testing.assert_almost_equal(
        utility_final_consume_all(random_choice, consumption, params, None), bequest
    )


@pytest.mark.parametrize(
    "consumption, mu, bequest_scale",
    list(product(CONSUMPTION_GRID, MU_GRID, BEQUEST_SCALE)),
)
def test_beqzest_marginal(consumption, mu, bequest_scale):
    params = {
        "mu": mu,
        "bequest_scale": bequest_scale,
    }
    random_choice = np.random.choice(np.array([0, 1, 2]))
    bequest = jax.jacfwd(utility_final_consume_all, argnums=1)(
        random_choice, consumption, params, None
    )
    np.testing.assert_almost_equal(
        marginal_utility_final_consume_all(random_choice, consumption, params, None),
        bequest,
    )
