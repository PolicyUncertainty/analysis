import jax
import numpy as np
from jax import numpy as jnp


def calc_early_retirement_pension_points(
    total_pension_points,
    retirement_age_difference,
    SRA_at_retirement,
    actual_retirement_age,
    experience_years,
    informed,
    education,
    health,
    sex,
    model_specs,
):
    """Calculate the pension points for early retirement for different paths. We implemented
    three ways to retire before the SRA:

        - Pension for long insured
        - Pension for very long insured
        - Disability pension

    We also check, which pension is best, given multiple are possible. There we use the
    following:
        - This function is only used, if the person freshly retires.
        - Disability pension is always better than pension for long insured, as the pension
          points fill up to the same level as if you would have worked until SRA and your
          deductions are limited to 3 years.
        - Pension for very long insured is always better than the pension for long insured.
          So we only check if very long insured path is better than disability pension.

    """
    # Check if the individual gets disability pension
    disability_pension_bool = health == model_specs["disabled_health_var"]
    # Check if the individual is eligible for very long insured pension
    very_long_insured_bool = check_very_long_insured(
        retirement_age_difference=retirement_age_difference,
        experience_years=experience_years,
        sex=sex,
        model_specs=model_specs,
    )

    # Additional pension points if disability retired.
    total_points_disability = calc_disability_pension_points(
        SRA_at_retirement=SRA_at_retirement,
        actual_retirement_age=actual_retirement_age,
        total_pension_points=total_pension_points,
    )

    # Penalty years. If disability pension(then limit to 3 years)
    penalty_years_disability = retirement_age_difference.clip(max=3)
    penalty_years = jax.lax.select(
        disability_pension_bool,
        on_true=penalty_years_disability,
        on_false=retirement_age_difference,
    )

    # Choose deduction factor according to information state
    early_retirement_penalty = (
        informed * model_specs["ERP"]
        + (1 - informed) * model_specs["uninformed_ERP"][education]
    )
    early_retirement_factor = 1 - early_retirement_penalty * penalty_years

    # Assign disability pension points if eligible. These are always higher in
    # case of early retirement.
    total_pension_points_checked = jax.lax.select(
        disability_pension_bool,
        on_true=total_points_disability,
        on_false=total_pension_points,
    )
    reduced_pension_points = early_retirement_factor * total_pension_points_checked

    # In case of disability these could be higher than total pension points now. So check
    # which is higher. In case of long insured pension, these are definetly lower.
    pension_points_very_long_insured = jnp.maximum(
        reduced_pension_points, total_pension_points
    )

    # If very long insured, we take the maximum from before.
    # Otherwise it is the reduced pension points.
    pension_points_early_retired = jax.lax.select(
        very_long_insured_bool,
        on_true=pension_points_very_long_insured,
        on_false=reduced_pension_points,
    )
    return pension_points_early_retired


def calc_disability_pension_points(
    total_pension_points,
    actual_retirement_age,
    SRA_at_retirement,
):
    """Calculate the disability pension points."""
    average_points_work_span = total_pension_points / (actual_retirement_age - 18)
    total_points_disability = (SRA_at_retirement - 18) * average_points_work_span
    return total_points_disability


def check_very_long_insured(
    retirement_age_difference, experience_years, sex, model_specs
):
    """Test if the individual qualifies for pension for very long insured
    (Rente besonders für langjährig Versicherte)."""

    experience_threshold = model_specs["experience_threshold_very_long_insured"][sex]
    enough_years = experience_years >= experience_threshold
    close_enough_to_SRA = retirement_age_difference <= 2
    return enough_years & close_enough_to_SRA


def retirement_age_long_insured(SRA, model_specs):
    """Everyone can retire 4 years before the SRA but must be at least at 63 y/o.
    That means that we assume 1) everyone qualifies for "Rente für langjährig Versicherte" and 2) that
    the threshhold for "Rente für langjährig Versicherte" moves with the SRA. "Rente für besonders langjährig
    Versicherte" only differs with respect to deductions. Not with respect to entry age. We introduce the
    lower bound of 63 as this is the current law, even for individuals with SRA below 67.
    """
    return np.maximum(
        SRA - model_specs["years_before_SRA_long_insured"],
        model_specs["min_long_insured_age"],
    )
