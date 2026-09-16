import jax
import jax.numpy as jnp
import numpy as np

from model_code.pension_system.experience_stock import (
    calc_pension_points_for_experience,
)


def define_experience_grid(specs):
    # Experience grid
    experience_grid = np.linspace(0, 1, 11)
    # Add very long insured threshold to experience grid and sort
    experience_grid = np.append(experience_grid, specs["very_long_insured_grid_points"])
    # Delete 0.5
    experience_grid = experience_grid[
        (~np.isclose(experience_grid, 0.5))
        & ~np.isclose(experience_grid, 0.6)
        & (~np.isclose(experience_grid, 0))
    ]

    experience_grid = np.sort(experience_grid)
    experience_grid[0] = 0
    experience_grid[1] = 0.15
    experience_grid[-3] = 0.85
    return jnp.asarray(experience_grid)


def build_experience_grid_by_period(specs):
    """Real-year, age-dependent experience grid for every period (not sex-specific yet).

    Built once here in NumPy, at spec-build time -- ``experience_grid_from_state``
    below is then a pure lookup into this table, not a per-call computation. The
    pooled ``[0, 1]`` grid (``define_experience_grid``) is multiplied by that
    period's real-year cap to get real credited-years, so dividing row ``p`` back
    by ``max_exps_period_working[p]`` reproduces ``define_experience_grid`` exactly
    -- this is a re-representation of the same pooled grid, not a different one.
    Periods past the working range (already-retired periods) reuse the last
    (maximum) entry, matching ``jnp.take(..., mode="clip")``'s behavior elsewhere
    in this module.
    """
    pooled_grid = np.asarray(define_experience_grid(specs))  # (n_nodes,), [0, 1]
    max_exps_period_working = np.asarray(specs["max_exps_period_working"])
    grid_max_by_period = np.take(
        max_exps_period_working, np.arange(specs["n_periods"]), mode="clip"
    )  # (n_periods,)
    table = grid_max_by_period[:, None] * pooled_grid[None, :]
    return jnp.asarray(table)  # (n_periods, n_nodes)


def experience_grid_from_state(period, model_specs):
    """Real-year, age-dependent experience grid dcegm evaluates per state-choice.

    Pure lookup into the precomputed ``experience_grid_by_period`` table (see
    ``build_experience_grid_by_period``).
    """
    return model_specs["experience_grid_by_period"][period]


def get_next_period_experience(
    period,
    lagged_choice,
    policy_state,
    sex,
    partner_state,
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
    pension_points = calc_pension_points_for_experience(
        period=period,
        experience_years=exp_years_last_period,
        sex=sex,
        partner_state=partner_state,
        education=education,
        policy_state=policy_state,
        informed=informed,
        health=health,
        model_specs=model_specs,
    )

    # If fresh retired, the experience function returns pension points. Now the value and policy function
    # are calculated on a pension point grid. We do not need experience any more.
    exp_years_this_period = jax.lax.select(
        fresh_retired, on_true=pension_points, on_false=exp_years_this_period
    )

    # Store on the grid's scale: real years while working, pension points
    # rescaled onto the grid's real-year range while retired.
    exp_scaled = scale_experience_years(
        experience_years=exp_years_this_period,
        period=period,
        is_retired=retired_this_period,
        model_specs=model_specs,
    )

    return {
        "experience": exp_scaled,
    }


def construct_experience_years(float_experience, period, is_retired, model_specs):
    """Recover real experience-years (working) or pension points (retired).

    The "experience" state is real credited-years directly while working (the
    grid itself is real-valued, see ``experience_grid_from_state``), so that
    branch is the identity. Retired states store pension points rescaled onto
    the same period's real-year range (see ``scale_experience_years``), so
    recovering the real pension points means undoing that rescaling. ``period``
    is kept for interface parity with callers across the codebase, even though
    the working branch no longer uses it.
    """
    grid_max = jnp.take(model_specs["max_exps_period_working"], period, mode="clip")
    pension_points = float_experience * model_specs["max_pp_retirement"] / grid_max
    return is_retired * pension_points + (1 - is_retired) * float_experience


def scale_experience_years(experience_years, period, is_retired, model_specs):
    """Inverse of ``construct_experience_years``: real years/pension points -> stored state."""
    grid_max = jnp.take(model_specs["max_exps_period_working"], period, mode="clip")
    scaled_pension_points = (
        experience_years * grid_max / model_specs["max_pp_retirement"]
    )
    return is_retired * scaled_pension_points + (1 - is_retired) * experience_years
