import jax
from jax import numpy as jnp

from model_code.wealth_and_budget.partner_income import calc_partner_income_after_ssc
from model_code.wealth_and_budget.pension_payments import calc_pensions_after_ssc
from model_code.wealth_and_budget.tax_and_ssc import calc_net_household_income
from model_code.wealth_and_budget.transfers import (
    calc_child_benefits,
    calc_unemployment_benefits,
)
from model_code.wealth_and_budget.wages import calc_labor_income_after_ssc


def budget_constraint(
    period,
    education,
    lagged_choice,  # d_{t-1}
    experience,
    sex,
    partner_state,
    health,
    asset_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    model_specs,
):
    assets_scaled = asset_end_of_previous_period * model_specs["wealth_unit"]
    # Recalculate experience
    max_exp_period = period + model_specs["max_exp_diffs_per_period"][period]
    experience_years = max_exp_period * experience

    # Calculate partner income
    partner_income_after_ssc, gross_partner_income = calc_partner_income_after_ssc(
        partner_state=partner_state,
        sex=sex,
        model_specs=model_specs,
        education=education,
        period=period,
    )

    # Income from lagged choice 0
    retirement_income_after_ssc = calc_pensions_after_ssc(
        experience_years=experience_years,
        sex=sex,
        education=education,
        model_specs=model_specs,
    )

    has_partner_int = (partner_state > 0).astype(int)

    # Income lagged choice 1
    unemployment_benefits = calc_unemployment_benefits(
        assets=assets_scaled,
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=model_specs,
    )

    # Income lagged choice 2
    labor_income_after_ssc, gross_labor_income = calc_labor_income_after_ssc(
        lagged_choice=lagged_choice,
        experience_years=experience_years,
        education=education,
        sex=sex,
        income_shock=income_shock_previous_period,
        model_specs=model_specs,
    )

    # Select relevant income
    # bools of last period decision: income is paid in following period!
    was_worker = lagged_choice >= 2
    was_retired = lagged_choice == 0

    # Aggregate over choice own income
    own_income_after_ssc = (
        was_worker * labor_income_after_ssc + was_retired * retirement_income_after_ssc
    )

    # Calculate total houshold net income
    total_net_income = calc_net_household_income(
        own_income=own_income_after_ssc,
        partner_income=partner_income_after_ssc,
        has_partner_int=has_partner_int,
        model_specs=model_specs,
    )
    child_benefits = calc_child_benefits(
        education=education,
        sex=sex,
        has_partner_int=has_partner_int,
        period=period,
        model_specs=model_specs,
    )

    total_income = jnp.maximum(total_net_income + child_benefits, unemployment_benefits)
    # calculate beginning of period wealth M_t
    assets_begin_of_period = (
        1 + model_specs["interest_rate"]
    ) * assets_scaled + total_income

    # death = health == model_specs["death_health_var"]
    # assets_begin_of_period = jax.lax.select(
    #     death, on_true=assets_scaled, on_false=assets_begin_of_period
    # )

    aux = {
        "net_hh_income": total_income / model_specs["wealth_unit"],
        "gross_hh_income": (gross_labor_income + gross_partner_income)
        / model_specs["wealth_unit"],
        "gross_own_income": gross_labor_income / model_specs["wealth_unit"],
    }

    return assets_begin_of_period / model_specs["wealth_unit"], aux
