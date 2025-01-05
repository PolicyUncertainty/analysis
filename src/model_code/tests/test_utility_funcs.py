from itertools import product

import jax
import numpy as np
import pytest
from model_code.utility.bequest_utility import marginal_utility_final_consume_all
from model_code.utility.bequest_utility import utility_final_consume_all
from model_code.utility.utility_functions import consumption_scale
from model_code.utility.utility_functions import disutility_work
from model_code.utility.utility_functions import inverse_marginal
from model_code.utility.utility_functions import marg_utility
from model_code.utility.utility_functions import utility_func
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs


jax.config.update("jax_enable_x64", True)

MU_GRID = np.linspace(0.1, 0.9, 2)
CONSUMPTION_GRID = np.linspace(10, 100, 3)
DISUTIL_UNEMPLOYED_GRID = np.linspace(0.1, 0.9, 2)
DISUTIL_WORK_GRID = np.linspace(0.1, 0.9, 2)
BEQUEST_SCALE = np.linspace(1, 4, 2)
PARTNER_STATE_GRIRD = np.array([0, 1, 2], dtype=int)
NB_CHILDREN_GRID = np.arange(0, 2, 0.5, dtype=int)
EDUCATION_GRID = np.array([0, 1], dtype=int)
HEALTH_GRID = np.array([0, 1], dtype=int)
PERIOD_GRID = np.arange(0, 15, 5, dtype=int)


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


@pytest.mark.parametrize(
    "partner_state, education, period",
    list(product(PARTNER_STATE_GRIRD, EDUCATION_GRID, PERIOD_GRID)),
)
def test_consumption_cale(partner_state, education, period, paths_and_specs):
    options = paths_and_specs[1]
    cons_scale = consumption_scale(partner_state, education, period, options)
    has_partner = int(partner_state > 0)
    nb_children = options["children_by_state"][0, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    np.testing.assert_almost_equal(cons_scale, np.sqrt(hh_size))


# """
@pytest.mark.parametrize(
    "consumption, partner_state, education, health, period, dis_util_work, dis_util_unemployed, mu",
    list(
        product(
            CONSUMPTION_GRID,
            PARTNER_STATE_GRIRD,
            EDUCATION_GRID,
            HEALTH_GRID,
            PERIOD_GRID,
            DISUTIL_WORK_GRID,
            DISUTIL_UNEMPLOYED_GRID,
            MU_GRID,
        )
    ),
)
def test_utility_func(
    consumption,
    partner_state,
    education,
    health,
    period,
    dis_util_work,
    dis_util_unemployed,
    mu,
    paths_and_specs,
):
    params = {
        "mu": mu,
        "dis_util_pt_work_good": dis_util_work + 1,
        "dis_util_pt_work_bad": dis_util_work,
        "dis_util_ft_work_good": dis_util_work + 1,
        "dis_util_ft_work_bad": dis_util_work,
        "dis_util_unemployed": dis_util_unemployed,
    }
    options = paths_and_specs[1]
    cons_scale = consumption_scale(partner_state, education, period, options)
    cons_utility = (consumption / cons_scale) ** (1 - mu) / (1 - mu)

    # Read out disutil params
    health_str = "good" * health + "bad" * (1 - health)
    disutil_unemployment = np.exp(-params[f"dis_util_unemployed_{health_str}"])
    dis_util_pt_work = np.exp(-params[f"dis_util_pt_work_{health_str}"])
    dis_util_ft_work = np.exp(-params[f"dis_util_ft_work_{health_str}"])

    np.testing.assert_almost_equal(
        utility_func(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            health=health,
            period=period,
            choice=1,
            params=params,
            options=options,
        ),
        cons_utility * disutil_unemployment ** (1 - mu),
    )

    np.testing.assert_almost_equal(
        utility_func(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            health=health,
            period=period,
            choice=2,
            params=params,
            options=options,
        ),
        cons_utility * dis_util_pt_work ** (1 - mu),
    )

    np.testing.assert_almost_equal(
        utility_func(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            health=health,
            period=period,
            choice=3,
            params=params,
            options=options,
        ),
        cons_utility * dis_util_ft_work ** (1 - mu),
    )


# """


# """


@pytest.mark.parametrize(
    "consumption, partner_state, education, health, period, dis_util_work, dis_util_unemployed, mu",
    list(
        product(
            CONSUMPTION_GRID,
            PARTNER_STATE_GRIRD,
            EDUCATION_GRID,
            HEALTH_GRID,
            PERIOD_GRID,
            DISUTIL_WORK_GRID,
            DISUTIL_UNEMPLOYED_GRID,
            MU_GRID,
        )
    ),
)
def test_marginal_utility(
    consumption,
    partner_state,
    education,
    health,
    period,
    dis_util_work,
    dis_util_unemployed,
    mu,
    paths_and_specs,
):
    options = paths_and_specs[1]
    params = {
        "mu": mu,
        "dis_util_pt_work_good": dis_util_work + 1,
        "dis_util_pt_work_bad": dis_util_work,
        "dis_util_ft_work_good": dis_util_work + 1,
        "dis_util_ft_work_bad": dis_util_work,
        "dis_util_unemployed": dis_util_unemployed,
    }

    random_choice = np.random.choice(np.array([0, 1, 2]))
    marg_util_jax = jax.jacfwd(utility_func, argnums=0)(
        consumption,
        partner_state,
        education,
        health,
        period,
        random_choice,
        params,
        options,
    )
    marg_util_model = marg_utility(
        consumption,
        partner_state,
        education,
        health,
        period,
        random_choice,
        params,
        options,
    )
    np.testing.assert_almost_equal(marg_util_jax, marg_util_model)


@pytest.mark.parametrize(
    "consumption, partner_state, education, health, period, dis_util_work, dis_util_unemployed, mu",
    list(
        product(
            CONSUMPTION_GRID,
            PARTNER_STATE_GRIRD,
            EDUCATION_GRID,
            HEALTH_GRID,
            PERIOD_GRID,
            DISUTIL_WORK_GRID,
            DISUTIL_UNEMPLOYED_GRID,
            MU_GRID,
        )
    ),
)
def test_inv_marginal_utility(
    consumption,
    partner_state,
    education,
    health,
    period,
    dis_util_work,
    dis_util_unemployed,
    mu,
    paths_and_specs,
):
    params = {
        "mu": mu,
        "dis_util_pt_work_good": dis_util_work + 1,
        "dis_util_pt_work_bad": dis_util_work,
        "dis_util_ft_work_good": dis_util_work + 1,
        "dis_util_ft_work_bad": dis_util_work,
        "dis_util_unemployed": dis_util_unemployed,
    }

    options = paths_and_specs[1]
    random_choice = np.random.choice(np.array([0, 1, 2]))
    marg_util = marg_utility(
        consumption,
        partner_state,
        education,
        health,
        period,
        random_choice,
        params,
        options,
    )
    np.testing.assert_almost_equal(
        inverse_marginal(
            marg_util,
            partner_state,
            education,
            health,
            period,
            random_choice,
            params,
            options,
        ),
        consumption,
    )


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
    np.testing.assert_almost_equal(
        utility_final_consume_all(consumption, params), bequest
    )


@pytest.mark.parametrize(
    "consumption, mu, bequest_scale",
    list(product(CONSUMPTION_GRID, MU_GRID, BEQUEST_SCALE)),
)
def test_bequest_marginal(consumption, mu, bequest_scale):
    params = {
        "mu": mu,
        "bequest_scale": bequest_scale,
    }
    bequest = jax.jacfwd(utility_final_consume_all, argnums=0)(consumption, params)
    np.testing.assert_almost_equal(
        marginal_utility_final_consume_all(consumption, params),
        bequest,
    )
