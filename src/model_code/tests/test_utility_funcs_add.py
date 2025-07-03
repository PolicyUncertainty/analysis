from itertools import product

import jax
import numpy as np
import pytest

from model_code.utility.bequest_utility import (
    marginal_utility_final_consume_all,
    utility_final_consume_all,
)
from model_code.utility.utility_functions_add import (
    consumption_scale,
    inverse_marginal_func,
    marginal_utility_function_alive,
    utility_func,
)
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

jax.config.update("jax_enable_x64", True)

MU_GRID = [1, 1.5]
CONSUMPTION_GRID = np.linspace(10, 100, 3)
disutil_UNEMPLOYED_GRID = np.linspace(0.1, 0.9, 2)
disutil_WORK_GRID = np.linspace(0.1, 0.9, 2)
BEQUEST_SCALE = np.linspace(1, 4, 2)
PARTNER_STATE_GRIRD = np.array([0, 1], dtype=int)
NB_CHILDREN_GRID = np.arange(0, 2, 0.5, dtype=int)
EDUCATION_GRID = np.array([0, 1], dtype=int)
HEALTH_GRID = np.array([0, 1, 2], dtype=int)
PERIOD_GRID = np.arange(0, 15, 5, dtype=int)
SEX_GRID = np.array([0, 1], dtype=int)


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


@pytest.mark.parametrize(
    "partner_state, sex, education, period",
    list(product(PARTNER_STATE_GRIRD, SEX_GRID, EDUCATION_GRID, PERIOD_GRID)),
)
def test_consumption_cale(partner_state, sex, education, period, paths_and_specs):
    model_specs = paths_and_specs[1]
    cons_scale = consumption_scale(
        partner_state=partner_state,
        sex=sex,
        education=education,
        period=period,
        model_specs=model_specs,
    )
    has_partner = int(partner_state > 0)
    nb_children = model_specs["children_by_state"][sex, education, has_partner, period]
    hh_size = 1 + has_partner + nb_children
    np.testing.assert_almost_equal(cons_scale, np.sqrt(hh_size))


# """
@pytest.mark.parametrize(
    "consumption, sex, partner_state, education, health, period, disutil_work, disutil_unemployed, mu",
    list(
        product(
            CONSUMPTION_GRID,
            SEX_GRID,
            PARTNER_STATE_GRIRD,
            EDUCATION_GRID,
            HEALTH_GRID,
            PERIOD_GRID,
            disutil_WORK_GRID,
            disutil_UNEMPLOYED_GRID,
            MU_GRID,
        )
    ),
)
def test_utility_func(
    consumption,
    sex,
    partner_state,
    education,
    health,
    period,
    disutil_work,
    disutil_unemployed,
    mu,
    paths_and_specs,
):
    params = {
        "mu": mu,
        "mu_bequest_low": mu + 1,
        "mu_bequest_high": mu,
        # Men
        "disutil_ft_work_good_men": disutil_work + 1,
        "disutil_ft_work_bad_men": disutil_work,
        "disutil_unemployed_bad_men": disutil_unemployed,
        "disutil_unemployed_good_men": disutil_unemployed,
        # Women
        "disutil_ft_work_high_good_women": disutil_work + 1,
        "disutil_ft_work_high_bad_women": disutil_work,
        "disutil_ft_work_low_good_women": disutil_work + 1,
        "disutil_ft_work_low_bad_women": disutil_work,
        "disutil_pt_work_high_good_women": disutil_work + 1,
        "disutil_pt_work_high_bad_women": disutil_work,
        "disutil_pt_work_low_good_women": disutil_work + 1,
        "disutil_pt_work_low_bad_women": disutil_work,
        "disutil_unemployed_high_women": disutil_unemployed,
        "disutil_unemployed_low_women": disutil_unemployed,
        "disutil_children_ft_work_low": 0.1,
        "disutil_children_ft_work_high": 0.1,
        "bequest_scale": 2,
    }

    model_specs = paths_and_specs[1]
    cons_scale = consumption_scale(
        partner_state=partner_state,
        sex=sex,
        education=education,
        period=period,
        model_specs=model_specs,
    )

    # Read out disutil params
    health_str = "good" if health == 0 else "bad"
    sex_str = "men" if sex == 0 else "women"
    edu_str = "low" if education == 0 else "high"

    if sex == 0:
        disutil_unemployment = params[f"disutil_unemployed_{health_str}_{sex_str}"]
        disutil_factor_ft_work = params[f"disutil_ft_work_{health_str}_{sex_str}"]

    else:
        disutil_unemployment = params[f"disutil_unemployed_{edu_str}_{sex_str}"]
        disutil_factor_ft_work = params[
            f"disutil_ft_work_{edu_str}_{health_str}_{sex_str}"
        ]

    if sex == 1:
        has_partner_int = int(partner_state > 0)
        nb_children = model_specs["children_by_state"][
            sex, education, has_partner_int, period
        ]
        disutil_factor_ft_work += (
            params["disutil_children_ft_work_high"] * nb_children * education
        )
        disutil_factor_ft_work += (
            params["disutil_children_ft_work_low"] * nb_children * (1 - education)
        )

    if mu == 1:
        utility_lambda = lambda disutil: np.log(consumption / cons_scale) - disutil
    else:
        utility_lambda = (
            lambda disutil: ((consumption / cons_scale) ** (1 - mu) - 1) / (1 - mu)
            - disutil
        )

    np.testing.assert_almost_equal(
        utility_func(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            health=health,
            sex=sex,
            period=period,
            choice=1,
            params=params,
            model_specs=model_specs,
        ),
        utility_lambda(disutil_unemployment),
    )
    if sex == 1:
        disutil_factor_pt_work = params[
            f"disutil_pt_work_{edu_str}_{health_str}_{sex_str}"
        ]

        np.testing.assert_almost_equal(
            utility_func(
                consumption=consumption,
                partner_state=partner_state,
                education=education,
                health=health,
                sex=sex,
                period=period,
                choice=2,
                params=params,
                model_specs=model_specs,
            ),
            utility_lambda(disutil_factor_pt_work),
        )

    np.testing.assert_almost_equal(
        utility_func(
            consumption=consumption,
            partner_state=partner_state,
            education=education,
            health=health,
            sex=sex,
            period=period,
            choice=3,
            params=params,
            model_specs=model_specs,
        ),
        utility_lambda(disutil_factor_ft_work),
    )


@pytest.mark.parametrize(
    "consumption, sex, partner_state, education, health, period, disutil_work, disutil_unemployed, mu",
    list(
        product(
            CONSUMPTION_GRID,
            SEX_GRID,
            PARTNER_STATE_GRIRD,
            EDUCATION_GRID,
            HEALTH_GRID,
            PERIOD_GRID,
            disutil_WORK_GRID,
            disutil_UNEMPLOYED_GRID,
            MU_GRID,
        )
    ),
)
def test_marginal_utility(
    consumption,
    sex,
    partner_state,
    education,
    health,
    period,
    disutil_work,
    disutil_unemployed,
    mu,
    paths_and_specs,
):
    model_specs = paths_and_specs[1]
    params = {
        "mu": mu,
        "mu_bequest_low": mu + 1,
        "mu_bequest_high": mu,
        # Men
        "disutil_ft_work_good_men": disutil_work + 1,
        "disutil_ft_work_bad_men": disutil_work,
        "disutil_unemployed_bad_men": disutil_unemployed,
        "disutil_unemployed_good_men": disutil_unemployed,
        # Women
        "disutil_ft_work_high_good_women": disutil_work + 1,
        "disutil_ft_work_high_bad_women": disutil_work,
        "disutil_ft_work_low_good_women": disutil_work + 1,
        "disutil_ft_work_low_bad_women": disutil_work,
        "disutil_pt_work_high_good_women": disutil_work + 1,
        "disutil_pt_work_high_bad_women": disutil_work,
        "disutil_pt_work_low_good_women": disutil_work + 1,
        "disutil_pt_work_low_bad_women": disutil_work,
        "disutil_unemployed_high_women": disutil_unemployed,
        "disutil_unemployed_low_women": disutil_unemployed,
        "disutil_children_ft_work_low": 0.1,
        "disutil_children_ft_work_high": 0.1,
        "bequest_scale": 2,
    }

    random_choice = np.random.choice(np.array([0, 1, 2]))
    marg_util_jax = jax.jacfwd(utility_func, argnums=0)(
        consumption,
        sex,
        partner_state,
        education,
        health,
        period,
        random_choice,
        params,
        model_specs,
    )
    marg_util_model = marginal_utility_function_alive(
        consumption=consumption,
        partner_state=partner_state,
        education=education,
        period=period,
        sex=sex,
        params=params,
        model_specs=model_specs,
    )
    np.testing.assert_almost_equal(marg_util_jax, marg_util_model)


@pytest.mark.parametrize(
    "consumption, sex, partner_state, education, health, period, disutil_work, disutil_unemployed, mu",
    list(
        product(
            CONSUMPTION_GRID,
            SEX_GRID,
            PARTNER_STATE_GRIRD,
            EDUCATION_GRID,
            HEALTH_GRID,
            PERIOD_GRID,
            disutil_WORK_GRID,
            disutil_UNEMPLOYED_GRID,
            MU_GRID,
        )
    ),
)
def test_inv_marginal_utility(
    consumption,
    sex,
    partner_state,
    education,
    health,
    period,
    disutil_work,
    disutil_unemployed,
    mu,
    paths_and_specs,
):
    params = {
        "mu": mu,
        "mu_bequest_low": mu + 1,
        "mu_bequest_high": mu,
        # Men
        "disutil_ft_work_good_men": disutil_work + 1,
        "disutil_ft_work_bad_men": disutil_work,
        "disutil_unemployed_bad_men": disutil_unemployed,
        "disutil_unemployed_good_men": disutil_unemployed,
        # Women
        "disutil_ft_work_high_good_women": disutil_work + 1,
        "disutil_ft_work_high_bad_women": disutil_work,
        "disutil_ft_work_low_good_women": disutil_work + 1,
        "disutil_ft_work_low_bad_women": disutil_work,
        "disutil_pt_work_high_good_women": disutil_work + 1,
        "disutil_pt_work_high_bad_women": disutil_work,
        "disutil_pt_work_low_good_women": disutil_work + 1,
        "disutil_pt_work_low_bad_women": disutil_work,
        "disutil_unemployed_high_women": disutil_unemployed,
        "disutil_unemployed_low_women": disutil_unemployed,
        "disutil_children_ft_work_low": 0.1,
        "disutil_children_ft_work_high": 0.1,
        "bequest_scale": 2,
    }

    model_specs = paths_and_specs[1]
    marg_util = marginal_utility_function_alive(
        consumption=consumption,
        partner_state=partner_state,
        education=education,
        sex=sex,
        period=period,
        params=params,
        model_specs=model_specs,
    )
    np.testing.assert_almost_equal(
        inverse_marginal_func(
            marginal_utility=marg_util,
            partner_state=partner_state,
            education=education,
            sex=sex,
            period=period,
            params=params,
            model_specs=model_specs,
        ),
        consumption,
    )


@pytest.mark.parametrize(
    "consumption, mu, education, bequest_scale",
    list(product(CONSUMPTION_GRID, MU_GRID, EDUCATION_GRID, BEQUEST_SCALE)),
)
def test_bequest(consumption, mu, education, bequest_scale):
    params = {
        "mu_bequest_low": mu + 1,
        "mu_bequest_high": mu,
        "bequest_scale": bequest_scale,
    }
    if education == 0:
        mu += 1
    if mu == 1:
        bequest = bequest_scale * np.log(consumption)
    else:
        bequest = bequest_scale * ((((consumption) ** (1 - mu)) - 1) / (1 - mu))
    np.testing.assert_almost_equal(
        utility_final_consume_all(consumption, education, params), bequest
    )


@pytest.mark.parametrize(
    "consumption, mu, education, bequest_scale",
    list(product(CONSUMPTION_GRID, MU_GRID, EDUCATION_GRID, BEQUEST_SCALE)),
)
def test_bequest_marginal(consumption, mu, education, bequest_scale):
    params = {
        "mu_bequest_low": mu + 1,
        "mu_bequest_high": mu,
        "bequest_scale": bequest_scale,
    }
    if education == 0:
        mu += 1
    bequest = jax.jacfwd(utility_final_consume_all, argnums=0)(
        consumption, education, params
    )
    np.testing.assert_almost_equal(
        marginal_utility_final_consume_all(consumption, education, params),
        bequest,
    )
