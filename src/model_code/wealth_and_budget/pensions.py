import jax.numpy as jnp
from model_code.wealth_and_budget.tax_and_ssc import calc_after_ssc_income_pensioneer


def calc_pensions_after_ssc(
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
    retirement_income = calc_after_ssc_income_pensioneer(
        retirement_income_gross, options
    )
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
    pension_deduction = (SRA_at_resolution - actual_retirement_age) * ERP
    pension_factor = 1 - pension_deduction

    # Pension point value by education and experience
    mean_wage_all = options["mean_wage"]
    gamma_0 = options["gamma_0"][education]
    gamma_1_plus_1 = options["gamma_1"][education] + 1
    total_pens_points = (
        (jnp.exp(gamma_0) / gamma_1_plus_1) * ((experience + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all

    retirement_income_gross = options["ppv"] * total_pens_points * pension_factor * 12
    return retirement_income_gross
