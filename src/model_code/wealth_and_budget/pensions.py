from model_code.wealth_and_budget.tax_and_transfers import calc_health_ltc_contr
from model_code.wealth_and_budget.tax_and_transfers import calc_inc_tax


def calc_pensions(
    experience,
    education,
    policy_state,
    retirement_age_id,
    options,
):
    # Retirement income
    retirement_income_gross = calc_gross_pension_income(
        experience=experience,
        education=education,
        policy_state=policy_state,
        retirement_age_id=retirement_age_id,
        options=options,
    )
    retirement_income = calc_net_income_pensions(retirement_income_gross, options)
    return retirement_income


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
    pension_point_value = options["pension_point_value_by_edu_exp"][
        education, experience
    ]

    retirement_income_gross = pension_point_value * experience * pension_factor * 12
    return retirement_income_gross


def calc_net_income_pensions(gross_income, options):
    gross_income_full = gross_income * options["wealth_unit"]
    ssc = calc_health_ltc_contr(gross_income_full)
    inc_tax = calc_inc_tax(gross_income_full - ssc)
    net_income = gross_income_full - inc_tax - ssc
    return net_income / options["wealth_unit"]
