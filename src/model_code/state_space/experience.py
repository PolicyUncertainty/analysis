import jax
from jax import numpy as jnp

from model_code.wealth_and_budget.pensions import (
    calc_experience_for_total_pension_points,
    calc_total_pension_points,
)


def get_next_period_experience(
    period,
    lagged_choice,
    policy_state,
    sex,
    education,
    experience,
    informed,
    partner_state,
    options,
):
    """Update experience based on lagged choice and period."""
    exp_years_last_period = construct_experience_years(
        experience=experience,
        period=period - 1,
        max_exp_diffs_per_period=options["max_exp_diffs_per_period"],
    )

    # Update if working part or full time
    exp_update = (lagged_choice == 3) + (lagged_choice == 2) * options[
        "exp_increase_part_time"
    ]
    exp_new_period = exp_years_last_period + exp_update

    # If retired, then we update experience according to the deduction function
    degenerate_state_id = options["n_policy_states"] - 1
    fresh_retired = (degenerate_state_id != policy_state) & (lagged_choice == 0)

    # Calculate experience with early retirement penalty
    adjusted_exp_years = calc_experience_years_for_pension_adjustment(
        period=period,
        experience_years=exp_years_last_period,
        sex=sex,
        education=education,
        policy_state=policy_state,
        informed=informed,
        partner_state=partner_state,
        options=options,
    )
    # Update if fresh retired
    exp_new_period = jax.lax.select(fresh_retired, adjusted_exp_years, exp_new_period)
    return (1 / (period + options["max_exp_diffs_per_period"][period])) * exp_new_period


def calc_experience_years_for_pension_adjustment(
    period,
    sex,
    experience_years,
    education,
    policy_state,
    informed,
    partner_state,
    options,
):
    """Calculate the reduced experience with early retirement penalty."""
    total_pension_points = calc_total_pension_points(
        education=education,
        experience_years=experience_years,
        sex=sex,
        options=options,
    )
    # retirement age is last periods age
    actual_retirement_age = options["start_age"] + period - 1
    # SRA at retirement, difference to actual retirement age and boolean for early retirement
    SRA_at_retirement = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    retirement_age_difference = jnp.abs(SRA_at_retirement - actual_retirement_age)
    early_retired_bool = actual_retirement_age < SRA_at_retirement

    # deduction factor for early  retirement
    early_retirement_penalty_informed = options["early_retirement_penalty"]
    early_retirement_penalty_uninformed = options[
        "uninformed_early_retirement_penalty"
    ][education]
    very_long_insured_test = test_very_long_insured(
        retirement_age_difference,
        experience_years,
        sex,
        education,
        partner_state,
        options,
    )
    early_retirement_penalty = (
        informed * early_retirement_penalty_informed
        + (1 - informed) * early_retirement_penalty_uninformed
    )
    early_retirement_penalty = (
        1
        - early_retirement_penalty
        * retirement_age_difference
        * (1 - very_long_insured_test)
    )

    # Total bonus for late retirement
    late_retirement_bonus = 1 + (
        options["late_retirement_bonus"] * retirement_age_difference
    )

    # Select bonus or penalty depending on age difference
    pension_factor = jax.lax.select(
        early_retired_bool, early_retirement_penalty, late_retirement_bonus
    )

    adjusted_pension_points = pension_factor * total_pension_points
    adjusted_experience_years = calc_experience_for_total_pension_points(
        total_pension_points=adjusted_pension_points,
        sex=sex,
        education=education,
        options=options,
    )
    return adjusted_experience_years


def construct_experience_years(experience, period, max_exp_diffs_per_period):
    """Experience and period can also be arrays."""
    return experience * (period + max_exp_diffs_per_period[period])


def test_very_long_insured(
    retirement_age_difference, experience_years, sex, education, partner_state, options
):
    """Test if the individual qualifies for pension for very long insured
    (Rente besonders für langjährig Versicherte)."""
    qualified_years = get_qualified_years(
        experience_years, sex, education, partner_state
    )
    enough_years = qualified_years >= 45
    close_enough_to_SRA = retirement_age_difference <= 2
    return enough_years * close_enough_to_SRA


def get_qualified_years(experience_years, sex, education, partner_state):
    """Calculate the qualified years for pension."""
    is_woman = sex == 1
    return experience_years + is_woman * 6
