import copy
from itertools import product

import numpy as np
import pytest

from model_code.state_space.experience import (
    get_next_period_experience,
    scale_experience_years,
)
from model_code.wealth_and_budget.budget_equation import budget_constraint
from model_code.wealth_and_budget.partner_income import calc_partner_income_after_ssc
from model_code.wealth_and_budget.tax_and_ssc import (
    calc_health_ltc_contr,
    calc_inc_tax_for_single_income,
    calc_pension_unempl_contr,
)
from model_code.wealth_and_budget.transfers import calc_unemployment_benefits
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

SAVINGS_GRID_UNEMPLOYED = np.linspace(10, 25, 7)
PARTNER_STATES = np.array([0, 1, 2], dtype=int)
PERIOD_GRID = np.arange(0, 65, 15, dtype=int)
OLD_AGE_PERIOD_GRID = np.arange(25, 43, 8, dtype=int)
VERY_OLD_AGE_PERIOD_GRID = np.arange(35, 43, 3, dtype=int)
EDUCATION_GRID = [0, 1]
SEX_GRID = [0, 1]


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


@pytest.mark.parametrize(
    "period, sex, partner_state, education, savings",
    list(
        product(
            PERIOD_GRID,
            SEX_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            SAVINGS_GRID_UNEMPLOYED,
        )
    ),
)
def test_budget_unemployed(
    period,
    sex,
    partner_state,
    education,
    savings,
    paths_and_specs,
):
    path_dict, specs = paths_and_specs

    specs_internal = copy.deepcopy(specs)

    exp_cont = scale_experience_years(
        experience_years=2,
        period=period,
        is_retired=False,
        model_specs=specs_internal,
    )

    wealth, _ = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        sex=sex,
        lagged_choice=1,
        experience=exp_cont,
        asset_end_of_previous_period=savings,
        income_shock_previous_period=0,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    has_partner = int(partner_state > 0)
    nb_children = specs["children_by_state"][sex, education, has_partner, period]
    income_partner, _, _ = calc_partner_income_after_ssc(
        partner_state=partner_state,
        sex=sex,
        model_specs=specs_internal,
        education=education,
        period=period,
    )
    split_factor = 1 + has_partner
    tax_partner = (
        calc_inc_tax_for_single_income(income_partner / split_factor, specs_internal)
        * split_factor
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
        (savings_scaled * (1 + specs["interest_rate"]) + income)
        / specs_internal["wealth_unit"],
    )


SAVINGS_GRID = np.linspace(8, 25, 4)
GAMMA_GRID = np.linspace(0.1, 0.9, 2)
EXP_GRID = np.linspace(10, 40, 10, dtype=int)
INCOME_SHOCK_GRID = np.linspace(-0.5, 0.5, 2)
WORKER_CHOICES = [2, 3]


@pytest.mark.parametrize(
    "working_choice, sex, period, partner_state ,education, gamma, income_shock, experience, savings",
    list(
        product(
            WORKER_CHOICES,
            SEX_GRID,
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
    sex,
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
    gamma_array = np.array([[gamma, gamma - 0.01], [gamma / 2, gamma / 2 - 0.01]])
    specs_internal["gamma_0"] = gamma_array
    specs_internal["gamma_1"] = gamma_array
    specs_internal["gamma_2"] = gamma_array - 1

    exp_cont = scale_experience_years(
        experience_years=experience,
        period=period,
        is_retired=False,
        model_specs=specs_internal,
    )

    wealth, aux_budget = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=working_choice,
        sex=sex,
        experience=exp_cont,
        asset_end_of_previous_period=savings,
        income_shock_previous_period=income_shock,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    hourly_wage = np.exp(
        gamma_array[sex, education]
        + gamma_array[sex, education] * experience
        + (gamma_array[sex, education] - 1) * experience**2
        + income_shock
    )
    if working_choice == 2:
        labor_income_year = (
            hourly_wage * specs_internal["av_annual_hours_pt"][sex, education]
        )
        min_wage_year = specs_internal["annual_min_wage_pt"][sex, education]
    else:
        labor_income_year = (
            hourly_wage * specs_internal["av_annual_hours_ft"][sex, education]
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
        assets=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    nb_children = specs_internal["children_by_state"][
        sex, education, has_partner_int, period
    ]
    child_benefits = nb_children * specs_internal["monthly_child_benefits"] * 12
    if partner_state == 0:
        tax_total = calc_inc_tax_for_single_income(income_after_ssc, specs_internal)
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
                sex, education, period
            ]

            sscs_partner = calc_health_ltc_contr(
                partner_income_year
            ) + calc_pension_unempl_contr(partner_income_year)
        else:
            partner_income_year = specs_internal["annual_partner_pension"][
                sex, education
            ]
            sscs_partner = calc_health_ltc_contr(partner_income_year)

        income_partner = partner_income_year - sscs_partner
        total_income_after_ssc = income_after_ssc + income_partner

        tax_toal = (
            calc_inc_tax_for_single_income(total_income_after_ssc / 2, specs_internal)
            * 2
        )
        total_net_income = total_income_after_ssc + child_benefits - tax_toal

        checked_income = np.maximum(total_net_income, unemployment_benefits)
        np.testing.assert_almost_equal(
            wealth,
            (savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income)
            / specs_internal["wealth_unit"],
        )


POLICY_STATE_GRID = np.linspace(0, 2, 3, dtype=int)


@pytest.mark.parametrize(
    "period, sex, partner_state ,education, savings, exp",
    list(
        product(
            VERY_OLD_AGE_PERIOD_GRID,
            SEX_GRID,
            PARTNER_STATES,
            EDUCATION_GRID,
            SAVINGS_GRID,
            EXP_GRID,
        )
    ),
)
def test_retiree(
    period,
    sex,
    partner_state,
    education,
    savings,
    exp,
    paths_and_specs,
):
    path_dict, specs_internal = paths_and_specs

    # exp is the pension points here
    scaled_pension_point_last_period = scale_experience_years(
        experience_years=exp,
        period=period - 1,
        is_retired=True,
        model_specs=specs_internal,
    )

    scaled_pension_points = get_next_period_experience(
        period=period,
        lagged_choice=np.array(0),
        policy_state=np.array(29),
        sex=sex,
        education=education,
        experience=scaled_pension_point_last_period,
        informed=0,
        health=2,
        model_specs=specs_internal,
    )
    # Check that experience does not get updated or added any penalty
    pension_points = scaled_pension_points * specs_internal["max_pp_retirement"]
    np.testing.assert_allclose(pension_points, exp)

    wealth, aux_budget = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=0,
        sex=sex,
        experience=scaled_pension_points,
        asset_end_of_previous_period=savings,
        income_shock_previous_period=0,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    pension_year = specs_internal["annual_pension_point_value"] * pension_points
    income_after_ssc = pension_year - calc_health_ltc_contr(pension_year)

    has_partner_int = (partner_state > 0).astype(int)
    unemployment_benefits = calc_unemployment_benefits(
        assets=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    nb_children = specs_internal["children_by_state"][
        sex, education, has_partner_int, period
    ]
    child_benefits = nb_children * specs_internal["monthly_child_benefits"] * 12
    if partner_state == 0:
        tax_total = calc_inc_tax_for_single_income(income_after_ssc, specs_internal)
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
                sex, education, period
            ]

            sscs_partner = calc_health_ltc_contr(
                partner_income_year
            ) + calc_pension_unempl_contr(partner_income_year)
        else:
            partner_income_year = specs_internal["annual_partner_pension"][
                sex, education
            ]
            sscs_partner = calc_health_ltc_contr(partner_income_year)

        income_partner = partner_income_year - sscs_partner
        total_income_after_ssc = income_after_ssc + income_partner

        tax_toal = (
            calc_inc_tax_for_single_income(total_income_after_ssc / 2, specs_internal)
            * 2
        )
        total_net_income = total_income_after_ssc + child_benefits - tax_toal

        checked_income = np.maximum(total_net_income, unemployment_benefits)
        scaled_wealth = (
            savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income
        )
        np.testing.assert_almost_equal(
            wealth, scaled_wealth / specs_internal["wealth_unit"]
        )


INFORMED_GRID = np.array([0, 1], dtype=int)
HEALTH_GRID = np.array([0, 1, 2], dtype=int)


@pytest.mark.parametrize(
    "period, sex ,education, savings, exp, policy_state, health, informed",
    list(
        product(
            OLD_AGE_PERIOD_GRID,
            SEX_GRID,
            EDUCATION_GRID,
            SAVINGS_GRID,
            EXP_GRID,
            POLICY_STATE_GRID,
            HEALTH_GRID,
            INFORMED_GRID,
        )
    ),
)
def test_fresh_retiree(
    period,
    sex,
    education,
    savings,
    exp,
    policy_state,
    informed,
    health,
    paths_and_specs,
):
    """In this test we assume that disability and non disability retirement is always possible.
    Even though in the model there are choice set restrictions, which is tested in state space.
    """
    path_dict, specs_internal = paths_and_specs

    # In this test, we set all to married, as this does not matter for the mechanic we test here,
    # i.e. the experience adjustment
    partner_state = np.array(1, dtype=int)

    actual_retirement_age = specs_internal["start_age"] + period - 1

    # Last the person was not retired. So exp is here really experience years
    exp_cont_prev = scale_experience_years(
        experience_years=exp,
        period=period - 1,
        is_retired=False,
        model_specs=specs_internal,
    )

    # This period the person retires, so lagged choice is 0. So this function returns
    # scaled pension points
    pension_points_scaled = get_next_period_experience(
        period=period,
        lagged_choice=0,
        policy_state=policy_state,
        sex=sex,
        education=education,
        experience=exp_cont_prev,
        informed=informed,
        health=health,
        model_specs=specs_internal,
    )

    wealth, aux_budget = budget_constraint(
        period=period,
        partner_state=partner_state,
        education=education,
        lagged_choice=0,
        sex=sex,
        experience=pension_points_scaled,
        asset_end_of_previous_period=savings,
        income_shock_previous_period=0,
        model_specs=specs_internal,
    )

    savings_scaled = savings * specs_internal["wealth_unit"]
    SRA_at_retirement = (
        specs_internal["min_SRA"] + policy_state * specs_internal["SRA_grid_size"]
    )
    retirement_age_difference = SRA_at_retirement - actual_retirement_age
    early_retirement = retirement_age_difference > 0

    # mean_wage_all = specs_internal["mean_hourly_ft_wage"][sex, education]
    # gamma_0 = specs_internal["gamma_0"][sex, education]
    # gamma_1_plus_1 = specs_internal["gamma_1"][sex, education] + 1
    # total_pens_points = (
    #     (np.exp(gamma_0) / gamma_1_plus_1) * ((exp + 1) ** gamma_1_plus_1 - 1)
    # ) / mean_wage_all

    exp_int = exp.astype(int)
    pp_exp_int = specs_internal["pp_for_exp_by_sex_edu"][sex, education, exp_int]

    exp_frac = exp - exp_int
    pp_difference = (
        specs_internal["pp_for_exp_by_sex_edu"][sex, education, exp_int + 1]
        - pp_exp_int
    )
    total_pens_points = pp_exp_int + exp_frac * pp_difference
    # We have only integer experience. So total_pension points should be equal to pp_exp_int
    np.testing.assert_array_almost_equal(total_pens_points, pp_exp_int)

    if early_retirement:
        if informed == 1:
            ERP = specs_internal["ERP"]
        else:
            ERP = specs_internal["uninformed_ERP"][education]

        if health == 2:
            retirement_age_difference = np.minimum(3, retirement_age_difference)
            average_points_work_span = total_pens_points / (actual_retirement_age - 18)
            disability_pens_points = average_points_work_span * (SRA_at_retirement - 18)

            reduced_pension_points = (
                1 - retirement_age_difference * ERP
            ) * disability_pens_points
        else:
            reduced_pension_points = (
                1 - retirement_age_difference * ERP
            ) * total_pens_points

        very_long_insured_bool = (retirement_age_difference <= 2) & (
            exp >= specs_internal["experience_threshold_very_long_insured"][sex]
        )
        if very_long_insured_bool:
            final_pension_points = np.maximum(total_pens_points, reduced_pension_points)
        else:
            final_pension_points = reduced_pension_points
    else:
        late_retirement_bonus = specs_internal["late_retirement_bonus"]
        pension_factor = 1 + np.abs(retirement_age_difference) * late_retirement_bonus
        final_pension_points = pension_factor * total_pens_points

    pension_year = specs_internal["annual_pension_point_value"] * final_pension_points
    income_after_ssc = pension_year - calc_health_ltc_contr(pension_year)

    has_partner_int = (partner_state > 0).astype(int)
    unemployment_benefits = calc_unemployment_benefits(
        assets=savings_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=specs_internal,
    )

    nb_children = specs_internal["children_by_state"][
        sex, education, has_partner_int, period
    ]
    child_benefits = nb_children * specs_internal["annual_child_benefits"]
    if partner_state == 0:
        tax_total = calc_inc_tax_for_single_income(income_after_ssc, specs_internal)
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
                sex, education, period
            ]

            sscs_partner = calc_health_ltc_contr(
                partner_income_year
            ) + calc_pension_unempl_contr(partner_income_year)
        else:
            partner_income_year = specs_internal["annual_partner_pension"][
                sex, education
            ]

            sscs_partner = calc_health_ltc_contr(partner_income_year)

        income_partner = partner_income_year - sscs_partner
        total_income_after_ssc = income_after_ssc + income_partner

        tax_toal = (
            calc_inc_tax_for_single_income(total_income_after_ssc / 2, specs_internal)
            * 2
        )
        total_net_income = total_income_after_ssc + child_benefits - tax_toal

        checked_income = np.maximum(total_net_income, unemployment_benefits)
        scaled_wealth = (
            savings_scaled * (1 + specs_internal["interest_rate"]) + checked_income
        )
        if early_retirement & (retirement_age_difference > 4) & (health != 2):
            pass
        else:
            np.testing.assert_almost_equal(
                wealth, scaled_wealth / specs_internal["wealth_unit"]
            )


def test_informed(
    paths_and_specs,
):

    path_dict, specs_internal = paths_and_specs

    period = 36

    exp_cont_prev = scale_experience_years(
        experience_years=45,
        period=period - 1,
        is_retired=False,
        model_specs=specs_internal,
    )

    exp_cont_uninf = get_next_period_experience(
        period=period,
        lagged_choice=np.array(0),
        policy_state=np.array(8),
        sex=0,
        education=0,
        experience=exp_cont_prev,
        informed=0,
        health=1,
        model_specs=specs_internal,
    )

    exp_cont_inf = get_next_period_experience(
        period=period,
        lagged_choice=np.array(0),
        policy_state=np.array(8),
        sex=0,
        education=0,
        experience=exp_cont_prev,
        informed=2,
        health=1,
        model_specs=specs_internal,
    )

    # Test if experience is the same
    np.testing.assert_allclose(exp_cont_inf, exp_cont_uninf)
