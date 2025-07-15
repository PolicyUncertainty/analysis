from model_code.pension_system.experience_stock import (
    calc_pension_points_form_experience,
)
from model_code.wealth_and_budget.tax_and_ssc import calc_after_ssc_income_pensioneer


def calc_pensions_after_ssc(
    experience_years,
    sex,
    education,
    model_specs,
):
    # Retirement income
    retirement_income_gross = calc_gross_pension_income(
        experience_years=experience_years,
        education=education,
        sex=sex,
        model_specs=model_specs,
    )
    retirement_income = calc_after_ssc_income_pensioneer(retirement_income_gross)
    return retirement_income


def calc_gross_pension_income(experience_years, sex, education, model_specs):
    """Calculate the gross pension income."""

    # Pension point value by education and experience
    pension_points = calc_pension_points_form_experience(
        education=education,
        sex=sex,
        experience_years=experience_years,
        model_specs=model_specs,
    )
    retirement_income_gross = model_specs["annual_pension_point_value"] * pension_points
    return retirement_income_gross
