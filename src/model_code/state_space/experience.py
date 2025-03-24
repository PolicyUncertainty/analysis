import jax
from jax import numpy as jnp

from model_code.state_space.choice_set import retirement_age_long_insured
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
    health,
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

    # Calculate experience in the case of fresh retirement
    # We track all deductions and bonuses of the retirement decision through an adjusted
    # experience stock
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

    # Check for fresh retirement. Not that we degenerate the policy state after the first period of
    # retirement and thus don't need to track when retired.
    degenerate_state_id = options["n_policy_states"] - 1
    fresh_retired = (degenerate_state_id != policy_state) & (lagged_choice == 0)
    exp_new_period = jax.lax.select(fresh_retired, adjusted_exp_years, exp_new_period)

    # Now scale between 0 and 1
    exp_scaled = (
        1 / (period + options["max_exp_diffs_per_period"][period])
    ) * exp_new_period
    return exp_scaled


def calc_experience_years_for_pension_adjustment(
    period,
    sex,
    experience_years,
    education,
    policy_state,
    informed,
    partner_state,
    health,
    options,
):
    """Calculate a new experience stock accounting for the pension adjustment.
    This function will only be used if the individual is fresh retired. So we
    can take this as a given here. Note, you can only retire if you are disabled or
    if you reached the long insured age.
    """
    # Start by calculating the type specific pension points(Endgeltpunkte)
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

    # Check if the individual gets disability pension
    # (remember in this function everybody is fresh retired)
    disability_pension_bool = health == options["disabled_health_var"]

    # Additional pension points if disability retired.
    age = period + options["start_age"]
    average_points_work_span = total_pension_points / (age - 18)
    total_points_disability = (SRA_at_retirement - 18) * average_points_work_span

    very_long_insured_bool = test_very_long_insured(
        retirement_age_difference=retirement_age_difference,
        experience_years=experience_years,
        sex=sex,
        education=education,
        partner_state=partner_state,
        options=options,
    )

    # Penalty years. First check if disability pension(then limit to 3 years)
    penalty_years_disability = retirement_age_difference.clip(max=3)
    penalty_years = jax.lax.select(
        disability_pension_bool,
        on_true=penalty_years_disability,
        on_false=retirement_age_difference,
    )

    # deduction factor for early  retirement
    early_retirement_penalty = (
        informed * options["ERP"]
        + (1 - informed) * options["uninformed_ERP"][education]
    )
    early_retirement_factor = 1 - early_retirement_penalty * penalty_years

    # Check which pension is best. Here we make use of the following.
    #     - Disability pension is always better than pension for long insured, as the pension
    #       points fill up to the same level as if you would have worked until SRA and your
    #       deductions are limited to 3 years.
    #     - Pension for very long insured is always better than the pension for long insured.
    #       So we really only need to check if very long insured path is better than disability
    #       pension

    # We first create reduced pension points.
    total_pension_points_checked = jax.lax.select(
        disability_pension_bool,
        on_true=total_points_disability,
        on_false=total_pension_points,
    )
    reduced_pension_points = early_retirement_factor * total_pension_points_checked

    # In case of disability these could be higher than total pension points now. So check
    # which is higher.
    pension_points_very_long_insured = jnp.maximum(
        reduced_pension_points, total_pension_points
    )

    # If very long insured, we take the maximum from before. Otherwise it is the reduced
    # pension points.
    pension_points_early_retired = jax.lax.select(
        very_long_insured_bool,
        on_true=pension_points_very_long_insured,
        on_false=reduced_pension_points,
    )

    # Total bonus for late retirement
    pension_points_late_retirement = (
        1 + (options["late_retirement_bonus"] * retirement_age_difference)
    ) * total_pension_points

    # Select bonus or penalty depending on age difference
    adjusted_pension_points = jax.lax.select(
        early_retired_bool,
        on_true=pension_points_early_retired,
        on_false=pension_points_late_retirement,
    )

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
        experience_years, sex, education, partner_state, options
    )
    enough_years = qualified_years >= 45
    close_enough_to_SRA = retirement_age_difference <= 2
    return enough_years * close_enough_to_SRA


def get_qualified_years(experience_years, sex, education, partner_state, options):
    """Calculate the qualified years for pension."""
    credited_periods_per_experience_point = options[
        "credited_periods_per_experience_point"
    ]
    qualified_years = credited_periods_per_experience_point[sex] * experience_years
    return qualified_years
