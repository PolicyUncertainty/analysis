from jax import numpy as jnp
from model_code.wealth_and_budget.tax_and_ssc import calc_after_ssc_income_worker


def calc_labor_income_after_ssc(experience, education, income_shock, options):
    # Gross labor income
    gross_labor_income = calculate_gross_labor_income(
        experience=experience,
        education=education,
        income_shock=income_shock,
        options=options,
    )
    net_labor_income = calc_after_ssc_income_worker(gross_labor_income, options)
    return net_labor_income


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
