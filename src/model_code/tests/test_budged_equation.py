import copy
from itertools import product

import numpy as np
import pytest
from model_code.state_space import get_next_period_experience
from model_code.wealth_and_budget.budget_equation import budget_constraint
from model_code.wealth_and_budget.partner_income import calc_partner_income_after_ssc
from model_code.wealth_and_budget.tax_and_ssc import calc_health_ltc_contr
from model_code.wealth_and_budget.tax_and_ssc import calc_inc_tax_for_single_income
from model_code.wealth_and_budget.tax_and_ssc import calc_pension_unempl_contr
from model_code.wealth_and_budget.transfers import calc_unemployment_benefits
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs


SAVINGS_GRID = np.linspace(10, 100, 3)
INTEREST_RATE_GRID = np.linspace(0.01, 0.1, 2)
PARTNER_STATES = np.array([0, 1, 2], dtype=int)
PERIOD_GRID = np.arange(0, 40, 10, dtype=int)
OLD_AGE_PERIOD_GRID = np.arange(33, 43, 1, dtype=int)
EDUCATION_GRID = [0, 1]


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


@pytest.mark.parametrize(
    "period, partner_state, education, savings, interest_rate",
    list(
        product(
            PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            SAVINGS_GRID,
            INTEREST_RATE_GRID,
        )
    ),
)
def test_budget_unemployed(
    period,
    partner_state,
    education,
    savings,
    interest_rate,
    paths_and_specs,
):
    path_dict, specs = paths_and_specs

    specs_internal = copy.deepcopy(specs)

    params = {"interest_rate": interest_rate}

    max_init_exp_period = period + specs_internal["max_init_experience"]
    exp_cont = 2 / max_init_exp_period

    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=1,
        experience=exp_cont,
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
        ]
        unemployment_benefits_children = (
            specs_internal["child_unemployment_benefits"] * nb_children
        )
        unemployment_benefits_housing = specs_internal[
            "unemployment_benefits_housing"
        ] * (1 + 0.5 * has_partner)
        unemployment_benefits_total = (
            unemployment_benefits
            + unemployment_benefits_children
            + unemployment_benefits_housing
        )
        income = np.maximum(
            unemployment_benefits_total * 12, net_partner_plus_child_benefits
        )
    else:
        income = net_partner_plus_child_benefits

    np.testing.assert_almost_equal(wealth, savings * (1 + interest_rate) + income)


GAMMA_GRID = np.linspace(0.1, 0.9, 3)
EXP_GRID = np.linspace(10, 30, 3, dtype=int)
INCOME_SHOCK_GRID = np.linspace(-0.5, 0.5, 2)


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
def test_budget_ft_worker(
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
    max_init_exp_period = period + specs_internal["max_init_experience"]
    exp_cont = experience / max_init_exp_period

    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=3,
        experience=exp_cont,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=income_shock,
        params=params,
        options=specs_internal,
    )
    hourly_wage = (
        np.exp(
            gamma_array[education]
            + gamma_array[education] * np.log(experience + 1)
            + income_shock
        )
        / specs_internal["wealth_unit"]
    )
    labor_income = hourly_wage * specs["av_hours_ft"]
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
    else:
        if partner_state == 1:
            partner_income_year = specs_internal["partner_wage"][education, period] * 12

            partner_income_scaled = partner_income_year * specs_internal["wealth_unit"]
            sscs_partner = calc_health_ltc_contr(
                partner_income_scaled
            ) + calc_pension_unempl_contr(partner_income_scaled)
        else:
            partner_income_year = specs_internal["partner_pension"][education] * 12
            partner_income_scaled = partner_income_year * specs_internal["wealth_unit"]
            sscs_partner = calc_health_ltc_contr(partner_income_scaled)

        income_partner = (
            partner_income_year - sscs_partner / specs_internal["wealth_unit"]
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


EXP_GRID = np.linspace(10, 30, 3, dtype=int)
POLICY_STATE_GRID = np.linspace(0, 2, 3, dtype=int)
RET_AGE_GRID = np.linspace(0, 2, 3, dtype=int)


@pytest.mark.parametrize(
    "period, partner_state ,education, interest_rate, savings, exp",
    list(
        product(
            OLD_AGE_PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            INTEREST_RATE_GRID,
            SAVINGS_GRID,
            EXP_GRID,
        )
    ),
)
def test_retiree(
    period,
    partner_state,
    education,
    interest_rate,
    savings,
    exp,
    paths_and_specs,
):
    path_dict, specs_internal = paths_and_specs

    params = {"interest_rate": interest_rate}
    max_init_exp_last_period = period + specs_internal["max_init_experience"] - 1
    exp_cont_last_period = exp / max_init_exp_last_period

    exp_cont = get_next_period_experience(
        period=period,
        lagged_choice=0,
        policy_state=29,
        education=education,
        experience=exp_cont_last_period,
        options=specs_internal,
    )

    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=0,
        experience=exp_cont,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=specs_internal,
    )
    mean_wage_all = specs_internal["mean_wage"]
    gamma_0 = specs_internal["gamma_0"][education]
    gamma_1_plus_1 = specs_internal["gamma_1"][education] + 1
    total_pens_points = (
        (np.exp(gamma_0) / gamma_1_plus_1) * ((exp + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all
    pension_year = specs_internal["ppv"] * total_pens_points * 12
    pension_scaled = pension_year * specs_internal["wealth_unit"]
    income_after_ssc = (
        pension_year
        - calc_health_ltc_contr(pension_scaled) / specs_internal["wealth_unit"]
    )

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
    else:
        if partner_state == 1:
            partner_income_year = specs_internal["partner_wage"][education, period] * 12

            partner_income_scaled = partner_income_year * specs_internal["wealth_unit"]
            sscs_partner = calc_health_ltc_contr(
                partner_income_scaled
            ) + calc_pension_unempl_contr(partner_income_scaled)
        else:
            partner_income_year = specs_internal["partner_pension"][education] * 12
            partner_income_scaled = partner_income_year * specs_internal["wealth_unit"]
            sscs_partner = calc_health_ltc_contr(partner_income_scaled)

        income_partner = (
            partner_income_year - sscs_partner / specs_internal["wealth_unit"]
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


@pytest.mark.parametrize(
    "period, partner_state ,education, interest_rate, savings, exp, policy_state",
    list(
        product(
            OLD_AGE_PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            INTEREST_RATE_GRID,
            SAVINGS_GRID,
            EXP_GRID,
            POLICY_STATE_GRID,
        )
    ),
)
def test_fresh_retiree(
    period,
    partner_state,
    education,
    interest_rate,
    savings,
    exp,
    policy_state,
    paths_and_specs,
):
    path_dict, specs_internal = paths_and_specs

    actual_retirement_age = specs_internal["start_age"] + period - 1

    params = {"interest_rate": interest_rate}
    max_init_exp_prev_period = period + specs_internal["max_init_experience"] - 1
    exp_cont_prev = exp / max_init_exp_prev_period

    exp_cont = get_next_period_experience(
        period=period,
        lagged_choice=0,
        policy_state=policy_state,
        education=education,
        experience=exp_cont_prev,
        options=specs_internal,
    )

    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=0,
        experience=exp_cont,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=0,
        params=params,
        options=specs_internal,
    )
    SRA_at_resolution = (
        specs_internal["min_SRA"] + policy_state * specs_internal["SRA_grid_size"]
    )
    deduction_factor = (SRA_at_resolution - actual_retirement_age) * specs_internal[
        "early_retirement_penalty"
    ]
    pension_factor = 1 - deduction_factor

    mean_wage_all = specs_internal["mean_wage"]
    gamma_0 = specs_internal["gamma_0"][education]
    gamma_1_plus_1 = specs_internal["gamma_1"][education] + 1
    total_pens_points = (
        (np.exp(gamma_0) / gamma_1_plus_1) * ((exp + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all

    pension_year = specs_internal["ppv"] * total_pens_points * pension_factor * 12
    pension_scaled = pension_year * specs_internal["wealth_unit"]
    income_after_ssc = (
        pension_year
        - calc_health_ltc_contr(pension_scaled) / specs_internal["wealth_unit"]
    )

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
    else:
        if partner_state == 1:
            partner_income_year = specs_internal["partner_wage"][education, period] * 12

            partner_income_scaled = partner_income_year * specs_internal["wealth_unit"]
            sscs_partner = calc_health_ltc_contr(
                partner_income_scaled
            ) + calc_pension_unempl_contr(partner_income_scaled)
        else:
            partner_income_year = specs_internal["partner_pension"][education] * 12
            partner_income_scaled = partner_income_year * specs_internal["wealth_unit"]
            sscs_partner = calc_health_ltc_contr(partner_income_scaled)

        income_partner = (
            partner_income_year - sscs_partner / specs_internal["wealth_unit"]
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
