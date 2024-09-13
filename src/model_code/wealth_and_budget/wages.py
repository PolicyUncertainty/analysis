from jax import numpy as jnp
from model_code.wealth_and_budget.tax_and_transfers import calc_health_ltc_contr
from model_code.wealth_and_budget.tax_and_transfers import calc_inc_tax
from model_code.wealth_and_budget.tax_and_transfers import calc_pension_unempl_contr


def calc_labor_income(experience, education, income_shock, options):
    # Gross labor income
    gross_labor_income = calculate_gross_labor_income(
        experience=experience,
        education=education,
        income_shock=income_shock,
        options=options,
    )
    net_labor_income = calc_net_income_working(gross_labor_income, options)
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


def calc_net_income_working(gross_income, options):
    gross_income_full = gross_income * options["wealth_unit"]
    ssc = calc_pension_unempl_contr(gross_income_full) + calc_health_ltc_contr(
        gross_income_full
    )
    inc_tax = calc_inc_tax(gross_income_full - ssc)
    net_income = gross_income_full - inc_tax - ssc
    return net_income / options["wealth_unit"]
