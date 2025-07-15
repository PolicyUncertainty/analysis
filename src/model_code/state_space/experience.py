import jax

from model_code.pension_system.experience_stock import (
    calc_experience_years_for_pension_adjustment,
)


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

    max_exp_diffs_per_period = model_specs["max_exp_diffs_per_period"]
    exp_years_last_period = construct_experience_years(
        experience=experience,
        period=period - 1,
        max_exp_diffs_per_period=max_exp_diffs_per_period,
    )

    # Update if working part or full time
    exp_update = (lagged_choice == 3) + (lagged_choice == 2) * model_specs[
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
        health=health,
        model_specs=model_specs,
    )

    # Check for fresh retirement. Not that we degenerate the policy state after the first period of
    # retirement and thus don't need to track when retired.
    degenerate_state_id = model_specs["n_policy_states"] - 1
    fresh_retired = (degenerate_state_id != policy_state) & (lagged_choice == 0)
    exp_new_period = jax.lax.select(fresh_retired, adjusted_exp_years, exp_new_period)

    # Now scale between 0 and 1
    exp_scaled = scale_experience_years(
        experience=exp_new_period,
        period=period,
        max_exp_diffs_per_period=max_exp_diffs_per_period,
    )
    return exp_scaled


def construct_experience_years(experience, period, max_exp_diffs_per_period):
    """Experience and period can also be arrays."""
    return experience * (period + max_exp_diffs_per_period[period])


def scale_experience_years(experience, period, max_exp_diffs_per_period):
    """Scale experience between 0 and 1."""
    return (1 / (period + max_exp_diffs_per_period[period])) * experience
