import jax
import jax.numpy as jnp
import numpy as np

from model_code.pension_system.experience_stock import (
    calc_pension_points_for_experience,
)

# Fixed skeleton on the normalized [0, 1] experience axis, shared by every type.
# 5 anchors pin the boundary and end spacing; 5 filler candidates fill the
# interior. Two fillers are dropped per sex to make room for that sex's
# very-long-insured bracket, so every sex ends up with the same 10 nodes.
_EXPERIENCE_ANCHORS = np.array([0.0, 0.15, 0.85, 0.9, 1.0])
_EXPERIENCE_FILLERS = np.array([0.3, 0.4, 0.5, 0.6, 0.7])


def build_experience_grid_by_sex(specs):
    """Per-sex experience grid on [0, 1]: one row per sex, all equal length.

    Splices in only that sex's very-long-insured bracket (the eligibility
    threshold and the half-year below it, the tightest bracket the half-year
    experience quantum allows) and drops the interior fillers nearest to it, so
    the discontinuity nodes replace crowding uniform nodes instead of clustering.
    Built once in NumPy at spec-build time; the grid dcegm evaluates per
    state-choice at solve time is ``experience_grid_from_state`` below.
    """
    vli = np.asarray(specs["very_long_insured_grid_points"])
    n_sexes = specs["n_sexes"]

    rows = []
    for sex in range(n_sexes):
        bracket = np.sort(vli[sex::n_sexes])  # this sex's [threshold - 0.5, threshold]
        dist = np.min(np.abs(_EXPERIENCE_FILLERS[:, None] - bracket[None, :]), axis=1)
        kept = _EXPERIENCE_FILLERS[np.sort(np.argsort(dist)[len(bracket) :])]
        rows.append(np.unique(np.concatenate([_EXPERIENCE_ANCHORS, kept, bracket])))

    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError(f"Per-sex experience grids differ in length: {lengths}")
    return jnp.asarray(np.stack(rows))


def experience_grid_from_state(sex, model_specs):
    """Grid dcegm evaluates per state-choice (jit-safe gather on ``sex``)."""
    return model_specs["experience_grid_by_sex"][sex]


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

    # Now scale between 0 and 1
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
    """Experience and period can also be arrays. We have to distinguish between the phases where individals are already
    longer retired or not."""
    # If period is past the last working period, then we take the maximum experience
    scale_not_retired = jnp.take(
        model_specs["max_exps_period_working"], period, mode="clip"
    )
    scale_retired = model_specs["max_pp_retirement"]
    scale = is_retired * scale_retired + (1 - is_retired) * scale_not_retired
    return float_experience * scale


def scale_experience_years(experience_years, period, is_retired, model_specs):
    """Scale experience between 0 and 1."""
    # If period is past the last working period, then we take the maximum experience
    scale_not_retired = jnp.take(
        model_specs["max_exps_period_working"], period, mode="clip"
    )
    scale_retired = model_specs["max_pp_retirement"]
    scale = is_retired * scale_retired + (1 - is_retired) * scale_not_retired
    return experience_years / scale
