from jax import numpy as jnp

def calc_unemployment_benefits(options, savings_end_of_previous_period):
    means_test = savings_end_of_previous_period < options["unemployment_wealth_thresh"]
    # Unemployment benefits
    unemployment_benefits = means_test * options["unemployment_benefits"] * 12
    return unemployment_benefits


def calc_net_income_pensions(gross_income):
    ssc = calc_health_ltc_contr(gross_income)
    inc_tax = calc_inc_tax(gross_income - ssc)
    net_income = gross_income - inc_tax - ssc
    return net_income


def calc_net_income_working(gross_income):
    ssc = calc_pension_unempl_contr(gross_income) + calc_health_ltc_contr(gross_income)
    inc_tax = calc_inc_tax(gross_income - ssc)
    net_income = gross_income - inc_tax - ssc
    return net_income


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
