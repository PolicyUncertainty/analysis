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


SAVINGS_GRID_UNEMPLOYED = np.linspace(10, 25, 5)
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
    "period, partner_state, education, savings",
    list(
        product(
            PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            SAVINGS_GRID_UNEMPLOYED,
        )
    ),
)
def test_budget_unemployed(
    period,
    partner_state,
    education,
    savings,
    paths_and_specs,
):
    path_dict, specs = paths_and_specs

    specs_internal = copy.deepcopy(specs)

    params = {"interest_rate": specs_internal["interest_rate"]}

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

    savings_scaled = savings * specs_internal["wealth_unit"]
    has_partner = int(partner_state > 0)
    nb_children = specs["children_by_state"][0, education, has_partner, period]
    income_partner = calc_partner_income_after_ssc(
        partner_state, specs_internal, education, period
    )
    split_factor = 1 + has_partner
    tax_partner = (
        calc_inc_tax_for_single_income(income_partner / split_factor) * split_factor
    )
    net_partner = income_partner - tax_partner
    net_partner_plus_child_benefits = (
        net_partner + nb_children * specs_internal["annual_child_benefits"]
    )

    unemployment_benefits = (1 + has_partner) * specs_internal[
        "annual_unemployment_benefits"
    ]
    unemployment_benefits_children = (
        specs_internal["annual_child_unemployment_benefits"] * nb_children
    )
    unemployment_benefits_housing = specs_internal[
        "annual_unemployment_benefits_housing"
    ] * (1 + 0.5 * has_partner)
    potential_unemployment_benefits = (
        unemployment_benefits
        + unemployment_benefits_children
        + unemployment_benefits_housing
    )

    means_test = savings_scaled < specs_internal["unemployment_wealth_thresh"]
    reduced_means_test_threshold = (
        specs_internal["unemployment_wealth_thresh"] + potential_unemployment_benefits
    )
    reduced_benefits_means_test = savings_scaled < reduced_means_test_threshold
    if means_test:
        income = np.maximum(
            potential_unemployment_benefits, net_partner_plus_child_benefits
        )
    elif ~means_test & reduced_benefits_means_test:
        reduced_unemployment_benefits = reduced_means_test_threshold - savings_scaled
        income = np.maximum(
            reduced_unemployment_benefits, net_partner_plus_child_benefits
        )
    else:
        income = net_partner_plus_child_benefits

    np.testing.assert_almost_equal(
        wealth,
        (savings_scaled * (1 + params["interest_rate"]) + income)
        / specs_internal["wealth_unit"],
    )


SAVINGS_GRID = np.linspace(8, 25, 3)
GAMMA_GRID = np.linspace(0.1, 0.9, 3)
EXP_GRID = np.linspace(10, 30, 3, dtype=int)
INCOME_SHOCK_GRID = np.linspace(-0.5, 0.5, 2)
WORKER_CHOICES = [2, 3]


@pytest.mark.parametrize(
    "working_choice, period, partner_state ,education, gamma, income_shock, experience, savings",
    list(
        product(
            WORKER_CHOICES,
            PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            GAMMA_GRID,
            INCOME_SHOCK_GRID,
            EXP_GRID,
            SAVINGS_GRID,
        )
    ),
)
def test_budget_worker(
    working_choice,
    period,
    partner_state,
    education,
    gamma,
    income_shock,
    experience,
    savings,
    paths_and_specs,
):
    path_dict, specs = paths_and_specs

    specs_internal = copy.deepcopy(specs)
    gamma_array = np.array([gamma, gamma - 0.01])
    specs_internal["gamma_0"] = gamma_array
    specs_internal["gamma_1"] = gamma_array

    params = {"interest_rate": specs_internal["interest_rate"]}
    max_init_exp_period = period + specs_internal["max_init_experience"]
    exp_cont = experience / max_init_exp_period

    wealth = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=working_choice,
        experience=exp_cont,
        savings_end_of_previous_period=savings,
        income_shock_previous_period=income_shock,
        params=params,
        options=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    hourly_wage = np.exp(
        gamma_array[education]
        + gamma_array[education] * np.log(experience + 1)
        + income_shock
    )
    if working_choice == 2:
        labor_income_year = (
            hourly_wage * specs_internal["av_annual_hours_pt"][education]
        )
        min_wage_year = specs_internal["annual_min_wage_pt"][education]
    else:
        labor_income_year = (
            hourly_wage * specs_internal["av_annual_hours_ft"][education]
        )
        min_wage_year = specs_internal["annual_min_wage_ft"]

    # Check against min wage
    if labor_income_year < min_wage_year:
        labor_income_year = min_wage_year

    income_scaled = labor_income_year
    sscs_worker = calc_health_ltc_contr(income_scaled) + calc_pension_unempl_contr(
        income_scaled
    )
    income_after_ssc = labor_income_year - sscs_worker

    has_partner_int = (partner_state > 0).astype(int)
    unemployment_benefits = calc_unemployment_benefits(
        savings_scaled, education, has_partner_int, period, specs_internal
    )

    nb_children = specs_internal["children_by_state"][
        0, education, partner_state, period
    ]
    child_benefits = nb_children * specs_internal["monthly_child_benefits"] * 12
    if partner_state == 0:
        tax_total = calc_inc_tax_for_single_income(income_after_ssc)
        total_net_income = income_after_ssc - tax_total + child_benefits
        checked_income = np.maximum(total_net_income, unemployment_benefits)
        np.testing.assert_almost_equal(
            wealth,
            (savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income)
            / specs_internal["wealth_unit"],
        )
    else:
        if partner_state == 1:
            partner_income_year = specs_internal["annual_partner_wage"][
                education, period
            ]

            sscs_partner = calc_health_ltc_contr(
                partner_income_year
            ) + calc_pension_unempl_contr(partner_income_year)
        else:
            partner_income_year = specs_internal["annual_partner_pension"][education]
            sscs_partner = calc_health_ltc_contr(partner_income_year)

        income_partner = partner_income_year - sscs_partner
        total_income_after_ssc = income_after_ssc + income_partner

        tax_toal = calc_inc_tax_for_single_income(total_income_after_ssc / 2) * 2
        total_net_income = total_income_after_ssc + child_benefits - tax_toal

        checked_income = np.maximum(total_net_income, unemployment_benefits)
        np.testing.assert_almost_equal(
            wealth,
            (savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income)
            / specs_internal["wealth_unit"],
        )


EXP_GRID = np.linspace(10, 30, 3, dtype=int)
POLICY_STATE_GRID = np.linspace(0, 2, 3, dtype=int)
RET_AGE_GRID = np.linspace(0, 2, 3, dtype=int)


@pytest.mark.parametrize(
    "period, partner_state ,education, savings, exp",
    list(
        product(
            OLD_AGE_PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            SAVINGS_GRID,
            EXP_GRID,
        )
    ),
)
def test_retiree(
    period,
    partner_state,
    education,
    savings,
    exp,
    paths_and_specs,
):
    path_dict, specs_internal = paths_and_specs

    params = {"interest_rate": specs_internal["interest_rate"]}
    max_init_exp_period = period + specs_internal["max_init_experience"]
    exp_cont_last_period = exp / (max_init_exp_period - 1)

    exp_cont = get_next_period_experience(
        period=period,
        lagged_choice=0,
        policy_state=29,
        education=education,
        experience=exp_cont_last_period,
        informed=0,
        options=specs_internal,
    )
    # Check that experience does not get updated or added any penalty
    np.testing.assert_allclose(exp_cont * max_init_exp_period, exp)

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

    savings_scaled = savings * specs_internal["wealth_unit"]

    mean_wage_all = specs_internal["mean_hourly_ft_wage"][education]
    gamma_0 = specs_internal["gamma_0"][education]
    gamma_1_plus_1 = specs_internal["gamma_1"][education] + 1
    total_pens_points = (
        (np.exp(gamma_0) / gamma_1_plus_1) * ((exp + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all
    pension_year = specs_internal["annual_pension_point_value"] * total_pens_points
    income_after_ssc = pension_year - calc_health_ltc_contr(pension_year)

    has_partner_int = (partner_state > 0).astype(int)
    unemployment_benefits = calc_unemployment_benefits(
        savings_scaled, education, has_partner_int, period, specs_internal
    )

    nb_children = specs_internal["children_by_state"][
        0, education, partner_state, period
    ]
    child_benefits = nb_children * specs_internal["monthly_child_benefits"] * 12
    if partner_state == 0:
        tax_total = calc_inc_tax_for_single_income(income_after_ssc)
        total_net_income = income_after_ssc - tax_total + child_benefits
        checked_income = np.maximum(total_net_income, unemployment_benefits)
        scaled_wealth = (
            savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income
        )
        np.testing.assert_almost_equal(
            wealth, scaled_wealth / specs_internal["wealth_unit"]
        )
    else:
        if partner_state == 1:
            partner_income_year = specs_internal["annual_partner_wage"][
                education, period
            ]

            sscs_partner = calc_health_ltc_contr(
                partner_income_year
            ) + calc_pension_unempl_contr(partner_income_year)
        else:
            partner_income_year = specs_internal["annual_partner_pension"][education]
            sscs_partner = calc_health_ltc_contr(partner_income_year)

        income_partner = partner_income_year - sscs_partner
        total_income_after_ssc = income_after_ssc + income_partner

        tax_toal = calc_inc_tax_for_single_income(total_income_after_ssc / 2) * 2
        total_net_income = total_income_after_ssc + child_benefits - tax_toal

        checked_income = np.maximum(total_net_income, unemployment_benefits)
        scaled_wealth = (
            savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income
        )
        np.testing.assert_almost_equal(
            wealth, scaled_wealth / specs_internal["wealth_unit"]
        )


INFORMED_GRID = np.array([0, 1], dtype=int)


@pytest.mark.parametrize(
    "period, partner_state ,education, savings, exp, policy_state, informed",
    list(
        product(
            OLD_AGE_PERIOD_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            SAVINGS_GRID,
            EXP_GRID,
            POLICY_STATE_GRID,
            INFORMED_GRID,
        )
    ),
)
def test_fresh_retiree(
    period,
    partner_state,
    education,
    savings,
    exp,
    policy_state,
    informed,
    paths_and_specs,
):
    path_dict, specs_internal = paths_and_specs

    actual_retirement_age = specs_internal["start_age"] + period - 1

    params = {"interest_rate": specs_internal["interest_rate"]}
    max_init_exp_prev_period = period + specs_internal["max_init_experience"] - 1
    exp_cont_prev = exp / max_init_exp_prev_period

    exp_cont = get_next_period_experience(
        period=period,
        lagged_choice=0,
        policy_state=policy_state,
        education=education,
        experience=exp_cont_prev,
        informed=informed,
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

    savings_scaled = savings * specs_internal["wealth_unit"]
    SRA_at_resolution = (
        specs_internal["min_SRA"] + policy_state * specs_internal["SRA_grid_size"]
    )
    retirement_age_difference = SRA_at_resolution - actual_retirement_age

    if retirement_age_difference > 0:
        if informed == 1:
            ERP = specs_internal["early_retirement_penalty"]
        else:
            ERP = specs_internal["uninformed_early_retirement_penalty"][education]
        pension_factor = 1 - retirement_age_difference * ERP
    else:
        late_retirement_bonus = specs_internal["late_retirement_bonus"]
        pension_factor = 1 + np.abs(retirement_age_difference) * late_retirement_bonus

    mean_wage_all = specs_internal["mean_hourly_ft_wage"][education]
    gamma_0 = specs_internal["gamma_0"][education]
    gamma_1_plus_1 = specs_internal["gamma_1"][education] + 1
    total_pens_points = (
        (np.exp(gamma_0) / gamma_1_plus_1) * ((exp + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all

    pension_year = (
        specs_internal["annual_pension_point_value"]
        * total_pens_points
        * pension_factor
    )
    income_after_ssc = pension_year - calc_health_ltc_contr(pension_year)

    has_partner_int = (partner_state > 0).astype(int)
    unemployment_benefits = calc_unemployment_benefits(
        savings_scaled, education, has_partner_int, period, specs_internal
    )

    nb_children = specs_internal["children_by_state"][
        0, education, partner_state, period
    ]
    child_benefits = nb_children * specs_internal["annual_child_benefits"]
    if partner_state == 0:
        tax_total = calc_inc_tax_for_single_income(income_after_ssc)
        total_net_income = income_after_ssc - tax_total + child_benefits
        checked_income = np.maximum(total_net_income, unemployment_benefits)
        scaled_wealth = (
            savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income
        )
        np.testing.assert_almost_equal(
            wealth, scaled_wealth / specs_internal["wealth_unit"]
        )
    else:
        if partner_state == 1:
            partner_income_year = specs_internal["annual_partner_wage"][
                education, period
            ]

            sscs_partner = calc_health_ltc_contr(
                partner_income_year
            ) + calc_pension_unempl_contr(partner_income_year)
        else:
            partner_income_year = specs_internal["annual_partner_pension"][education]

            sscs_partner = calc_health_ltc_contr(partner_income_year)

        income_partner = partner_income_year - sscs_partner
        total_income_after_ssc = income_after_ssc + income_partner

        tax_toal = calc_inc_tax_for_single_income(total_income_after_ssc / 2) * 2
        total_net_income = total_income_after_ssc + child_benefits - tax_toal

        checked_income = np.maximum(total_net_income, unemployment_benefits)
        scaled_wealth = (
            savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income
        )
        np.testing.assert_almost_equal(
            wealth, scaled_wealth / specs_internal["wealth_unit"]
        )
