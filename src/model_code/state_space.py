import jax
import jax.numpy as jnp
import numpy as np
from model_code.wealth_and_budget.pensions import (
    calc_experience_for_total_pension_points,
)
from model_code.wealth_and_budget.pensions import calc_total_pension_points


def create_state_space_functions():
    return {
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_experience": get_next_period_experience,
        "sparsity_condition": sparsity_condition,
    }


def sparsity_condition(
    period,
    lagged_choice,
    sex,
    informed,
    health,
    partner_state,
    policy_state,
    education,
    options,
):
    start_age = options["start_age"]
    max_ret_age = options["max_ret_age"]
    min_ret_age_state_space = options["min_ret_age"]
    # Generate last period, because only here are death states
    last_period = options["n_periods"] - 1
    # Degenerated policy state
    degenerate_policy_state = options["n_policy_states"] - 1

    age = start_age + period
    if (sex == 0) & (lagged_choice == 2):
        return False
    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age_state_space) & (lagged_choice == 0):
        return False
    # After the maximum retirement age, you must be retired.
    elif (age > max_ret_age) & (lagged_choice != 0) & (health != 2):
        return False
    else:
        # Now turn to the states, where it is decided by the value of an exogenous
        # state if it is valid or not. For invalid states we provide a proxy child state
        if health == 2:
            # Lead all states with death to last period death states
            # with job offer 0 (not relevant for bequest). You could be in principle
            # die upon retirement for which we need informed and policy state
            state_proxy = {
                "period": last_period,
                "lagged_choice": lagged_choice,
                "education": education,
                "health": health,
                "informed": informed,
                "sex": sex,
                "partner_state": partner_state,
                "job_offer": 0,
                "policy_state": policy_state,
            }
            return state_proxy
        elif (age <= max_ret_age + 1) and (lagged_choice == 0):
            # If retirement is already chosen we proxy all states to job offer 0.
            # Until age max_ret_age + 1 the individual could also be freshly retired
            # so we check if the policy state is degenerated. If so, we proxy to
            # informed states only
            if policy_state == degenerate_policy_state:
                state_proxy = {
                    "period": period,
                    "lagged_choice": lagged_choice,
                    "education": education,
                    "health": health,
                    "informed": 1,
                    "sex": sex,
                    "partner_state": partner_state,
                    "job_offer": 0,
                    "policy_state": policy_state,
                }
            else:
                state_proxy = {
                    "period": period,
                    "lagged_choice": lagged_choice,
                    "education": education,
                    "health": health,
                    "informed": informed,
                    "partner_state": partner_state,
                    "sex": sex,
                    "job_offer": 0,
                    "policy_state": policy_state,
                }
            return state_proxy
        elif age > max_ret_age + 1:
            # If age is larger than max_ret_age + 1, the individual can only be longer retired.
            # We can degenerate the policy state to and also only keep informed.
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "education": education,
                "health": health,
                "informed": 1,
                "sex": sex,
                "partner_state": partner_state,
                "job_offer": 0,
                "policy_state": options["n_policy_states"] - 1,
            }
            return state_proxy
        else:
            return True


def state_specific_choice_set(
    period, lagged_choice, sex, policy_state, job_offer, health, options
):
    age = period + options["start_age"]
    SRA_pol_state = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(SRA_pol_state, options)

    # If somebody is death, we assign a dummy choice set of [0]
    if health == 2:
        return np.array([0])
    # retirement is absorbing
    elif lagged_choice == 0:
        return np.array([0])
    # Check if the person is not in the voluntary retirement range.
    elif age < min_ret_age_pol_state:
        if job_offer == 0:
            return np.array([1])
        else:
            if sex == 0:
                return np.array([1, 3])
            else:
                return np.array([1, 2, 3])
    elif age >= options["max_ret_age"]:
        return np.array([0])
    else:
        if age >= SRA_pol_state:
            if job_offer == 0:
                return np.array([0])
            else:
                if sex == 0:
                    return np.array([0, 3])
                else:
                    return np.array([0, 2, 3])
        else:
            if job_offer == 0:
                return np.array([0, 1])
            else:
                if sex == 0:
                    return np.array([0, 1, 3])
                else:
                    return np.array([0, 1, 2, 3])


def apply_retirement_constraint_for_SRA(SRA, options):
    return np.maximum(SRA - options["ret_years_before_SRA"], 63)


def get_next_period_experience(
    period, lagged_choice, policy_state, sex, education, experience, informed, options
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
    experience_years_with_penalty = calc_experience_years_for_pension_adjustment(
        period=period,
        experience_years=exp_years_last_period,
        sex=sex,
        education=education,
        policy_state=policy_state,
        informed=informed,
        options=options,
    )
    # Update if fresh retired
    exp_new_period = jax.lax.select(
        fresh_retired, experience_years_with_penalty, exp_new_period
    )
    return (1 / (period + options["max_exp_diffs_per_period"][period])) * exp_new_period


def calc_experience_years_for_pension_adjustment(
    period, sex, experience_years, education, policy_state, informed, options
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
    early_retirement_penalty = (
        informed * early_retirement_penalty_informed
        + (1 - informed) * early_retirement_penalty_uninformed
    )
    early_retirement_penalty = 1 - early_retirement_penalty * retirement_age_difference

    # Total bonus for late retirement
    late_retirement_bonus = 1 + (
        options["late_retirement_bonus"] * retirement_age_difference
    )

    # Select bonus or penalty depending on age difference
    pension_factor = jax.lax.select(
        early_retired_bool, early_retirement_penalty, late_retirement_bonus
    )

    adjusted_pension_points = pension_factor * total_pension_points
    reduced_experience_years = calc_experience_for_total_pension_points(
        total_pension_points=adjusted_pension_points,
        sex=sex,
        education=education,
        options=options,
    )
    return reduced_experience_years


def construct_experience_years(experience, period, max_exp_diffs_per_period):
    """Experience and period can also be arrays."""
    return experience * (period + max_exp_diffs_per_period[period])
