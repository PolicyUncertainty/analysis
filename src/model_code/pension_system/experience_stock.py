import jax
from jax import numpy as jnp

from model_code.pension_system.early_retirement_paths import (
    calc_early_retirement_pension_points,
)


def calc_experience_years_for_pension_adjustment(
    period,
    sex,
    experience_years,
    education,
    policy_state,
    informed,
    health,
    model_specs,
):
    """Calculate a new experience stock accounting for the pension adjustment.
    This function will only be used if the individual is fresh retired. So we
    can take this as a given here. Note, you can only retire if you are disabled or
    if you reached the long insured age.
    """
    # Start by calculating the type specific pension points(Endgeltpunkte)
    total_pension_points = calc_pension_points_form_experience(
        education=education,
        experience_years=experience_years,
        sex=sex,
        model_specs=model_specs,
    )
    # retirement age is last periods age
    actual_retirement_age = model_specs["start_age"] + period - 1
    # SRA at retirement, difference to actual retirement age and boolean for early retirement
    SRA_at_retirement = (
        model_specs["min_SRA"] + policy_state * model_specs["SRA_grid_size"]
    )
    retirement_age_difference = jnp.abs(SRA_at_retirement - actual_retirement_age)
    early_retired_bool = actual_retirement_age < SRA_at_retirement

    # Pension points for early retirement
    pension_points_early_retired = calc_early_retirement_pension_points(
        total_pension_points=total_pension_points,
        retirement_age_difference=retirement_age_difference,
        SRA_at_retirement=SRA_at_retirement,
        actual_retirement_age=actual_retirement_age,
        experience_years=experience_years,
        informed=informed,
        education=education,
        health=health,
        sex=sex,
        model_specs=model_specs,
    )

    # Total bonus for late retirement
    pension_points_late_retirement = (
        1 + (model_specs["late_retirement_bonus"] * retirement_age_difference)
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
        model_specs=model_specs,
    )
    return adjusted_experience_years


def calc_pension_points_form_experience(education, sex, experience_years, model_specs):
    """Calculate the total pension point for the working live.

    We normalize by the mean wage of the whole population. The punishment for early
    retirement is already in the experience.

    """
    mean_wage_all = model_specs["mean_hourly_ft_wage"][sex, education]
    gamma_0 = model_specs["gamma_0"][sex, education]
    gamma_1_plus_1 = model_specs["gamma_1"][sex, education] + 1
    total_pens_points = (
        (jnp.exp(gamma_0) / gamma_1_plus_1)
        * ((experience_years + 1) ** gamma_1_plus_1 - 1)
    ) / mean_wage_all
    return total_pens_points


def calc_experience_for_total_pension_points(
    total_pension_points, sex, education, model_specs
):
    """Calculate the experience for a given total pension points."""
    mean_wage_all = model_specs["mean_hourly_ft_wage"][sex, education]
    gamma_0 = model_specs["gamma_0"][sex, education]
    gamma_1_plus_1 = model_specs["gamma_1"][sex, education] + 1
    return (
        (total_pension_points * mean_wage_all * gamma_1_plus_1 / jnp.exp(gamma_0) + 1)
        ** (1 / gamma_1_plus_1)
    ) - 1
