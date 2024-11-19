import jax.numpy as jnp
from model_code.wealth_and_budget.tax_and_ssc import calc_after_ssc_income_pensioneer


def calc_pensions_after_ssc(
    experience_years,
    education,
    options,
):
    # Retirement income
    retirement_income_gross = calc_gross_pension_income(
        experience_years=experience_years,
        education=education,
        options=options,
    )
    retirement_income = calc_after_ssc_income_pensioneer(retirement_income_gross)
    return retirement_income


def calc_gross_pension_income(experience_years, education, options):
    """Calculate the gross pension income."""

    # Pension point value by education and experience
    total_pension_points = calc_total_pension_points(
        education=education, experience_years=experience_years, options=options
    )
    retirement_income_gross = (
        options["annual_pension_point_value"] * total_pension_points
    )
    return retirement_income_gross


def calc_total_pension_points(education, experience_years, options):
    """Calculate the total pension point for the working live.

    We normalize by the mean wage of the whole population. The punishment for early
    retirement is already in the experience.

    """
    mean_wage_all = options["mean_hourly_ft_wage"][education]
    gamma_0 = options["gamma_0"][education]
    gamma_1_plus_1 = options["gamma_1"][education] + 1
    total_pens_points = (
        (jnp.exp(gamma_0) / gamma_1_plus_1)
        * ((experience_years + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all
    return total_pens_points


def calc_experience_for_total_pension_points(total_pension_points, education, options):
    """Calculate the experience for a given total pension points."""
    mean_wage_all = options["mean_hourly_ft_wage"][education]
    gamma_0 = options["gamma_0"][education]
    gamma_1_plus_1 = options["gamma_1"][education] + 1
    return (
        (total_pension_points * mean_wage_all * gamma_1_plus_1 / jnp.exp(gamma_0) + 1)
        ** (1 / gamma_1_plus_1)
    ) - 1
