import copy
from itertools import product

import numpy as np
import pytest
from model_code.wealth_and_budget.budget_equation import budget_constraint
from model_code.wealth_and_budget.partner_income import calc_partner_income_after_ssc
from model_code.wealth_and_budget.tax_and_ssc import calc_health_ltc_contr
from model_code.wealth_and_budget.tax_and_ssc import calc_inc_tax_for_single_income
from model_code.wealth_and_budget.tax_and_ssc import calc_pension_unempl_contr
from model_code.wealth_and_budget.transfers import calc_unemployment_benefits
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs


SAVINGS_GRID = np.linspace(10, 100, 5)
INTEREST_RATE_GRID = np.linspace(0.01, 0.1, 2)
PARTNER_STATES = np.array([0, 1, 2], dtype=int)
PERIOD_GRID = np.arange(0, 50, 10, dtype=int)
EDUCATION_GRID = [0, 1]

BENEFITS_GRID = np.linspace(10, 100, 5)


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


@pytest.mark.parametrize(
    "period, partner_state, education, unemployment_benefits, savings, interest_rate",
    list(
        product(
            PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            BENEFITS_GRID,
            SAVINGS_GRID,
            INTEREST_RATE_GRID,
        )
    ),
)
def test_budget_unemployed(
    period,
    partner_state,
    education,
    unemployment_benefits,
    savings,
    interest_rate,
    paths_and_specs,
):
    path_dict, specs = paths_and_specs

    specs_internal = copy.deepcopy(specs)
    specs_internal["unemployment_benefits"] = unemployment_benefits

    params = {"interest_rate": interest_rate}
    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=0,
        experience=30,
        policy_state=0,
        retirement_age_id=0,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=specs_internal,
    )
    has_partner = int(partner_state > 0)
    nb_children = specs["children_by_state"][0, education, has_partner, period]
    income_partner = calc_partner_income_after_ssc(
        partner_state, specs_internal, education, period
    )
    split_factor = 1 + has_partner
    tax_partner = (
        calc_inc_tax_for_single_income(income_partner / split_factor, specs_internal)
        * split_factor
    )
    net_partner = income_partner - tax_partner
    net_partner_plus_child_benefits = (
        net_partner + nb_children * specs_internal["child_benefit"] * 12
    )

    if savings < specs_internal["unemployment_wealth_thresh"]:
        unemployment_benefits = (1 + has_partner) * specs_internal[
            "unemployment_benefits"
        ] + specs_internal["child_unemployment_benefits"] * nb_children
        income = np.maximum(unemployment_benefits * 12, net_partner_plus_child_benefits)
    else:
        income = net_partner_plus_child_benefits

    np.testing.assert_almost_equal(wealth, savings * (1 + interest_rate) + income)


GAMMA_GRID = np.linspace(0.1, 0.9, 3)
EXP_GRID = np.linspace(10, 30, 3, dtype=int)
INCOME_SHOCK_GRID = np.linspace(-0.5, 0.5, 3)


@pytest.mark.parametrize(
    "period, partner_state ,education, gamma, income_shock, experience, interest_rate, savings",
    list(
        product(
            PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            GAMMA_GRID,
            INCOME_SHOCK_GRID,
            EXP_GRID,
            INTEREST_RATE_GRID,
            SAVINGS_GRID,
        )
    ),
)
def test_budget_worker(
    period,
    partner_state,
    education,
    gamma,
    income_shock,
    experience,
    interest_rate,
    savings,
    paths_and_specs,
):
    path_dict, specs = paths_and_specs

    specs_internal = copy.deepcopy(specs)
    gamma_array = np.array([gamma, gamma - 0.01])
    specs_internal["gamma_0"] = gamma_array
    specs_internal["gamma_1"] = gamma_array

    params = {"interest_rate": interest_rate}

    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=1,
        experience=experience,
        policy_state=0,
        retirement_age_id=0,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=income_shock,
        params=params,
        options=specs_internal,
    )
    labor_income = (
        np.exp(
            gamma_array[education]
            + gamma_array[education] * np.log(experience + 1)
            + income_shock
        )
        / specs_internal["wealth_unit"]
    )
    if labor_income < specs_internal["min_wage"]:
        labor_income = specs_internal["min_wage"]

    labor_income_year = labor_income * 12
    income_scaled = labor_income_year * specs_internal["wealth_unit"]
    sscs_worker = calc_health_ltc_contr(income_scaled) + calc_pension_unempl_contr(
        income_scaled
    )
    income_after_ssc = labor_income_year - sscs_worker / specs_internal["wealth_unit"]

    has_partner_int = (partner_state > 0).astype(int)
    unemployment_benefits = calc_unemployment_benefits(
        savings, education, has_partner_int, period, specs_internal
    )

    nb_children = specs_internal["children_by_state"][
        0, education, partner_state, period
    ]
    child_benefits = nb_children * specs_internal["child_benefit"] * 12
    if partner_state == 0:
        tax_total = calc_inc_tax_for_single_income(income_after_ssc, specs_internal)
        total_net_income = income_after_ssc - tax_total + child_benefits
        checked_income = np.maximum(total_net_income, unemployment_benefits)
        np.testing.assert_almost_equal(
            wealth, savings * (1 + interest_rate) + checked_income
        )

    elif partner_state == 1:
        partner_wage_year = specs_internal["partner_wage"][education, period] * 12

        partner_wage_scaled = partner_wage_year * specs_internal["wealth_unit"]
        sscs_partner = calc_health_ltc_contr(
            partner_wage_scaled
        ) + calc_pension_unempl_contr(partner_wage_scaled)

        income_partner = (
            partner_wage_year - sscs_partner / specs_internal["wealth_unit"]
        )
        total_income_after_ssc = income_after_ssc + income_partner

        tax_toal = (
            calc_inc_tax_for_single_income(total_income_after_ssc / 2, specs_internal)
            * 2
        )
        total_net_income = total_income_after_ssc + child_benefits - tax_toal

        checked_income = np.maximum(total_net_income, unemployment_benefits)
        np.testing.assert_almost_equal(
            wealth, savings * (1 + interest_rate) + checked_income
        )

    else:
        pass


#
# EXP_GRID = np.linspace(10, 30, 3, dtype=int)
# POLICY_STATE_GRID = np.linspace(0, 2, 3, dtype=int)
# RET_AGE_GRID = np.linspace(0, 2, 3, dtype=int)
#
#
# @pytest.mark.parametrize(
#     "interest_rate, savings, exp, policy_state, ret_age_id",
#     list(
#         product(
#             INTEREST_RATE_GRID, SAVINGS_GRID, EXP_GRID, POLICY_STATE_GRID, RET_AGE_GRID
#         )
#     ),
# )
# def test_retiree(
#     interest_rate,
#     savings,
#     exp,
#     policy_state,
#     ret_age_id,
#     paths_and_specs,
# ):
#     education = 0
#
#     path_dict, specs = paths_and_specs
#
#     actual_retirement_age = specs["min_ret_age"] + ret_age_id
#
#     params = {"interest_rate": interest_rate}
#     wealth = budget_constraint(
#         education=education,
#         lagged_choice=2,
#         experience=exp,
#         policy_state=policy_state,
#         retirement_age_id=ret_age_id,
#         savings_end_of_previous_period=savings,
#         income_shock_previous_period=0,
#         params=params,
#         options=specs,
#     )
#     pension_point_value = specs["pension_point_value_by_edu_exp"][education, exp]
#     SRA_at_resolution = specs["min_SRA"] + policy_state * specs["SRA_grid_size"]
#     pension_factor = (
#         1
#         - (actual_retirement_age - SRA_at_resolution)
#         * specs["early_retirement_penalty"]
#     )
#     retirement_income = calc_net_income_pensions(
#         pension_point_value * pension_factor * exp * 12, specs
#     )
#     if savings < specs["unemployment_wealth_thresh"]:
#         retirement_income = np.maximum(
#             retirement_income, specs["unemployment_benefits"] * 12
#         )
#
#     np.testing.assert_almost_equal(
#         wealth, savings * (1 + interest_rate) + retirement_income
#     )
