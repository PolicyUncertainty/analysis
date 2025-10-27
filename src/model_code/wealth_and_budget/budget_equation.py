import jax
from jax import numpy as jnp

from model_code.state_space.experience import construct_experience_years
from model_code.wealth_and_budget.alg_1 import calc_potential_alg_1
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
    alg_1_claim,
    asset_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    model_specs,
):
    assets_scaled = asset_end_of_previous_period * model_specs["wealth_unit"]

    age = model_specs["start_age"] + period
    # Recalculate experience
    experience_years = construct_experience_years(
        float_experience=experience,
        period=period,
        is_retired=lagged_choice == 0,
        model_specs=model_specs,
    )

    # Calculate partner income
    partner_income_after_ssc, gross_partner_wage, gross_partner_pension = (
        calc_partner_income_after_ssc(
            partner_state=partner_state,
            sex=sex,
            model_specs=model_specs,
            education=education,
            period=period,
        )
    )

    # Income from lagged choice 0. Here the experience is already transformed into pension points,
    # as we only track those in retirement.
    retirement_income_after_ssc, gross_retirement_income = calc_pensions_after_ssc(
        pension_points=experience_years,
        model_specs=model_specs,
    )

    has_partner_int = (partner_state > 0).astype(int)

    # Income lagged choice 1
    unemployment_benefits, own_unemployemnt_benefits = calc_unemployment_benefits(
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
        age=age,
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

    # Alg 1 is tax free as well. Calc potential alg_1 for unemployed
    potential_alg_1 = calc_potential_alg_1(
        age=age,
        education=education,
        sex=sex,
        experience_years=experience_years,
        model_specs=model_specs,
    )
    was_unemployed = lagged_choice == 1
    alg_1 = (alg_1_claim > 0) * was_unemployed * potential_alg_1

    total_net_income += child_benefits + alg_1

    total_income = jnp.maximum(total_net_income, unemployment_benefits)
    interest_rate = model_specs["interest_rate"]
    interest = interest_rate * assets_scaled
    income_plus_interest = total_income + interest
    # calculate beginning of period wealth M_t
    assets_begin_of_period = assets_scaled + income_plus_interest

    # death = health == model_specs["death_health_var"]
    # assets_begin_of_period = jax.lax.select(
    #     death, on_true=assets_scaled, on_false=assets_begin_of_period
    # )

    aux = {
        "net_hh_income": income_plus_interest / model_specs["wealth_unit"],
        "hh_net_income_wo_interest": total_income / model_specs["wealth_unit"],
        "interest": interest / model_specs["wealth_unit"],
        "joint_gross_labor_income": (gross_labor_income + gross_partner_wage)
        / model_specs["wealth_unit"],
        "joint_gross_retirement_income": (
            gross_partner_pension + gross_retirement_income
        )
        / model_specs["wealth_unit"],
        "gross_labor_income": gross_labor_income / model_specs["wealth_unit"],
        "gross_retirement_income": gross_retirement_income / model_specs["wealth_unit"],
    }

    return assets_begin_of_period / model_specs["wealth_unit"], aux
