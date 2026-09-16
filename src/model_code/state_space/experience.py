import jax
import jax.numpy as jnp

from model_code.pension_system.experience_stock import (
    calc_pension_points_for_experience,
)


def experience_grid_from_state(period, sex, model_specs):
    """Sex- and age-specific experience grid dcegm evaluates per state-choice.

    A single sex-specific real-year axis per period, used by both working and
    retired states: retired states reinterpret the same axis as pension points
    (rescaled onto it, see ``scale_experience_years``). This is required by dcegm's
    shared-child consistency check -- a not-yet-retired individual who chooses
    retirement and an already-retired individual transition to the same child state,
    so they must use the same grid; a separate retired grid is therefore not
    representable here. Pure lookup into the table precomputed at spec-build time
    (see ``build_experience_grid_working_by_sex_period``).
    """
    return model_specs["experience_grid_working_by_sex_period"][sex, period]


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
        sex=sex,
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
        sex=sex,
        is_retired=retired_this_period,
        model_specs=model_specs,
    )

    return {
        "experience": exp_scaled,
    }


def _experience_grid_cap(period, sex, model_specs):
    """The sex- and age-specific working grid cap (its top node).

    ``period`` may be ``-1`` (see ``get_next_period_experience``'s handling of
    period 0), so it is clipped into range before indexing. Works elementwise for
    scalar or array ``sex``/``period``.
    """
    cap_by_sex_period = model_specs[
        "experience_grid_cap_by_period"
    ]  # (n_sexes, n_periods)
    n_periods = cap_by_sex_period.shape[1]
    period_clipped = jnp.clip(period, 0, n_periods - 1)
    return cap_by_sex_period[sex, period_clipped]


def construct_experience_years(float_experience, period, sex, is_retired, model_specs):
    """Recover real experience-years (working) or pension points (retired).

    The "experience" state is real credited-years directly while working (the
    grid itself is real-valued, see ``experience_grid_from_state``), so that
    branch is the identity. Retired states store pension points rescaled onto
    the same sex-period real-year axis (see ``scale_experience_years``), so
    recovering the real pension points means undoing that rescaling. The rescaling
    constant is the working grid's own per-(sex, period) cap (its top node), so
    retired stored values land within the grid.
    """
    grid_max = _experience_grid_cap(period, sex, model_specs)
    pension_points = float_experience * model_specs["max_pp_retirement"] / grid_max
    return is_retired * pension_points + (1 - is_retired) * float_experience


def scale_experience_years(experience_years, period, sex, is_retired, model_specs):
    """Inverse of ``construct_experience_years``: real years/pension points -> stored state."""
    grid_max = _experience_grid_cap(period, sex, model_specs)
    scaled_pension_points = (
        experience_years * grid_max / model_specs["max_pp_retirement"]
    )
    return is_retired * scaled_pension_points + (1 - is_retired) * experience_years
