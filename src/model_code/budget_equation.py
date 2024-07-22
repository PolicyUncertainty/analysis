import jax.numpy as jnp
import numpy as np


def budget_constraint(
    education,
    lagged_choice,  # d_{t-1}
    experience,
    policy_state,  # current applicable SRA identifyer
    retirement_age_id,
    savings_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    options,
):
    # gross pension income
    retirement_income_gross = calc_gross_pension_income(
        experience=experience,
        education=education,
        policy_state=policy_state,
        retirement_age_id=retirement_age_id,
        options=options,
    )
    retirement_income = calc_net_income_pensions(retirement_income_gross, options)

    means_test = savings_end_of_previous_period < options["unemployment_wealth_thresh"]

    # Unemployment benefits
    unemployment_benefits = means_test * options["unemployment_benefits"] * 12

    # Gross labor income
    gross_labor_income = calculate_gross_labor_income(
        experience=experience,
        education=education,
        income_shock=income_shock_previous_period,
        options=options,
    )
    net_labor_income = calc_net_income_working(gross_labor_income, options)

    # bools of last period decision: income is payed in following period!
    was_worker = lagged_choice == 1
    was_retired = lagged_choice == 2

    income = jnp.maximum(
        was_worker * net_labor_income + was_retired * retirement_income,
        unemployment_benefits,
    )

    # calculate beginning of period wealth M_t
    wealth = (1 + params["interest_rate"]) * savings_end_of_previous_period + income

    return wealth


def create_savings_grid():
    """Create a saving grid with sections."""
    section_1 = np.arange(start=0, stop=10, step=0.5)  # 20
    section_2 = np.arange(start=10, stop=50, step=1)  # 40
    section_3 = np.arange(start=50, stop=100, step=5)  # 10
    section_4 = np.arange(start=100, stop=500, step=20)  # 20
    section_5 = np.arange(start=500, stop=1000, step=100)  # 5
    savings_grid = np.concatenate(
        [section_1, section_2, section_3, section_4, section_5]
    )
    return savings_grid


def calculate_gross_labor_income(experience, education, income_shock, options):
    """Calculate the gross labor income.

    As we estimate the wage equation outside of the model, we fetch the experience
    returns from options.

    """
    gamma_0 = options["gamma_0"][education]
    gamma_1 = options["gamma_1"][education]
    labor_income = jnp.exp(gamma_0 + gamma_1 * jnp.log(experience + 1) + income_shock)

    labor_income_min_checked = (
        jnp.maximum(labor_income / options["wealth_unit"], options["min_wage"]) * 12
    )

    return labor_income_min_checked


def calc_gross_pension_income(
    experience, education, policy_state, retirement_age_id, options
):
    """Calculate the gross pension income."""
    # generate actual retirement age and SRA at resolution
    SRA_at_resolution = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    actual_retirement_age = options["min_ret_age"] + retirement_age_id

    # deduction (bonus) factor for early (late) retirement
    ERP = options["early_retirement_penalty"]
    pension_factor = 1 - (actual_retirement_age - SRA_at_resolution) * ERP

    # Pension point value by education and experience
    pension_point_value = (
        options["adjustment_factor_by_edu_and_exp"][education, experience]
        * options["pension_point_value"]
    )

    retirement_income_gross = pension_point_value * experience * pension_factor * 12
    return retirement_income_gross


def calc_net_income_pensions(gross_income, options):
    gross_income_full = gross_income * options["wealth_unit"]
    ssc = calc_health_ltc_contr(gross_income_full)
    inc_tax = calc_inc_tax(gross_income_full - ssc)
    net_income = gross_income_full - inc_tax - ssc
    return net_income / options["wealth_unit"]


def calc_net_income_working(gross_income, options):
    gross_income_full = gross_income * options["wealth_unit"]
    ssc = calc_pension_unempl_contr(gross_income_full) + calc_health_ltc_contr(
        gross_income_full
    )
    inc_tax = calc_inc_tax(gross_income_full - ssc)
    net_income = gross_income_full - inc_tax - ssc
    return net_income / options["wealth_unit"]


def calc_inc_tax(gross_income):
    """Parameters from 2010 gettsim params."""
    thresholds = [
        8004,
        13469,
        52881,
        250730,
    ]

    rates = [0.14, 0.2397, 0.42, 0.45]

    # In bracket 0 no taxes are paid
    poss_tax_bracket_0 = 0.0

    # In bracket 1 taxes are paid on income above the threshold
    poss_tax_bracket_1 = rates[0] * (gross_income - thresholds[0])
    tax_above_1 = (thresholds[1] - thresholds[0]) * rates[0]

    # In bracket 2 taxes are paid on income above the threshold and the tax paid in
    # bracket 1
    poss_tax_bracket_2 = (rates[1] * (gross_income - thresholds[1])) + tax_above_1
    tax_above_2 = (thresholds[2] - thresholds[1]) * rates[1]

    # In bracket 3 taxes are paid on income above the threshold and the tax paid in
    # brackets 1+2
    poss_tax_bracket_3 = (
        rates[2] * (gross_income - thresholds[2]) + tax_above_2 + tax_above_1
    )
    tax_above_3 = (thresholds[3] - thresholds[2]) * rates[2]

    # In bracket 4 taxes are paid on income above the threshold and the tax paid in
    # brackets 1+2+3
    poss_tax_bracket_4 = (
        rates[3] * (gross_income - thresholds[3])
        + tax_above_1
        + tax_above_2
        + tax_above_3
    )

    # Check in which bracket the income falls and calculate the tax
    in_bracket_0 = gross_income < thresholds[0]
    in_bracket_1 = (gross_income >= thresholds[0]) & (gross_income < thresholds[1])
    in_bracket_2 = (gross_income >= thresholds[1]) & (gross_income < thresholds[2])
    in_bracket_3 = (gross_income >= thresholds[2]) & (gross_income < thresholds[3])
    in_bracket_4 = gross_income >= thresholds[3]

    income_tax = (
        in_bracket_0 * poss_tax_bracket_0
        + in_bracket_1 * poss_tax_bracket_1
        + in_bracket_2 * poss_tax_bracket_2
        + in_bracket_3 * poss_tax_bracket_3
        + in_bracket_4 * poss_tax_bracket_4
    )
    return income_tax


def calc_pension_unempl_contr(gross_income):
    contribution_threshold = 5500 * 12
    rate = 0.113
    # calculate pension contribution
    pension_contr = jnp.minimum(gross_income, contribution_threshold) * rate
    return pension_contr


def calc_health_ltc_contr(gross_income):
    contribution_threshold = 3750 * 12
    rate = 0.08
    # calculate social security contribution
    health_contr = jnp.minimum(gross_income, contribution_threshold) * rate
    return health_contr
