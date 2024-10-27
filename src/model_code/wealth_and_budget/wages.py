from jax import numpy as jnp
from model_code.wealth_and_budget.tax_and_ssc import calc_after_ssc_income_worker


def calc_labor_income_after_ssc(
    lagged_choice, experience, education, income_shock, options
):
    # Gross labor income
    gross_labor_income = calculate_gross_labor_income(
        lagged_choice=lagged_choice,
        experience=experience,
        education=education,
        income_shock=income_shock,
        options=options,
    )
    labor_income_after_ssc = calc_after_ssc_income_worker(gross_labor_income, options)
    return labor_income_after_ssc


def calculate_gross_labor_income(
    lagged_choice, experience, education, income_shock, options
):
    """Calculate the gross labor income.

    As we estimate the wage equation outside of the model, we fetch the experience
    returns from options.

    """
    gamma_0 = options["gamma_0"][education]
    gamma_1 = options["gamma_1"][education]
    hourly_wage = jnp.exp(gamma_0 + gamma_1 * jnp.log(experience + 1) + income_shock)

    # Part time choice
    pt_work = lagged_choice == 2
    ft_work = lagged_choice == 3

    average_hours = (
        options["av_annual_hours_pt"][education] * pt_work
        + options["av_annual_hours_ft"][education] * ft_work
    )
    labour_income = hourly_wage * average_hours

    yearly_min_wage = options["min_wage"] * 12
    labor_income_min_checked = jnp.maximum(
        labour_income / options["wealth_unit"], yearly_min_wage
    )
    return labor_income_min_checked
