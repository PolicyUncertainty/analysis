import jax
import jax.numpy as jnp
import numpy as np

from model_code.pension_system.experience_stock import (
    calc_experience_years_for_pension_adjustment,
)


def define_experience_grid(specs):
    # Experience grid
    experience_grid = np.linspace(0, 1, 11)
    # Add very long insured threshold to experience grid and sort
    # experience_grid = np.append(experience_grid, specs["very_long_insured_grid_points"])
    # # Delete 0.5
    # experience_grid = experience_grid[~np.isclose(experience_grid, 0.5)]
    experience_grid = jnp.sort(experience_grid)
    return experience_grid


def get_next_period_experience(
    period,
    lagged_choice,
    policy_state,
    sex,
    education,
    experience,
    informed,
    health,
    model_specs,
):
    """Update experience based on lagged choice and period."""
    # Check if already longer retired. If it is not degenerated you could have not been
    # retired last period.
    degenerate_state_id = model_specs["n_policy_states"] - 1
    retired_last_period = degenerate_state_id == policy_state
    retired_this_period = lagged_choice == 0
    # Fresh retirement means not retired last period and retired this period.
    fresh_retired = ~retired_last_period & retired_this_period

    last_period = period - 1
    # If period is 0, then last period is also 0.
    last_period = last_period * (period != 0) + (period == 0) * (-1)

    exp_years_last_period = construct_experience_years(
        float_experience=experience,
        period=last_period,
        is_retired=retired_last_period,
        model_specs=model_specs,
    )

    # Update if working part or full time
    exp_update = (lagged_choice == 3) + (lagged_choice == 2) * model_specs[
        "exp_increase_part_time"
    ]
    exp_years_this_period = exp_years_last_period + exp_update

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
        health=health,
        model_specs=model_specs,
    )

    exp_years_this_period = jax.lax.select(
        fresh_retired, adjusted_exp_years, exp_years_this_period
    )

    # Now scale between 0 and 1
    exp_scaled = scale_experience_years(
        experience_years=exp_years_this_period,
        period=period,
        is_retired=retired_this_period,
        model_specs=model_specs,
    )

    return exp_scaled


def construct_experience_years(float_experience, period, is_retired, model_specs):
    """Experience and period can also be arrays. We have to distinguish between the phases where individals are already
    longer retired or not."""
    # If period is past the last working period, then we take the maximum experience
    scale_not_retired = jnp.take(
        model_specs["max_exps_period_working"], period, mode="clip"
    )
    scale_retired = model_specs["max_exp_retirement"]
    scale = is_retired * scale_retired + (1 - is_retired) * scale_not_retired
    return float_experience * scale


def scale_experience_years(experience_years, period, is_retired, model_specs):
    """Scale experience between 0 and 1."""
    # If period is past the last working period, then we take the maximum experience
    scale_not_retired = jnp.take(
        model_specs["max_exps_period_working"], period, mode="clip"
    )
    scale_retired = model_specs["max_exp_retirement"]
    scale = is_retired * scale_retired + (1 - is_retired) * scale_not_retired
    return experience_years / scale
