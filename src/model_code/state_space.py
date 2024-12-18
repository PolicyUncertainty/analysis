import jax
import jax.numpy as jnp
import numpy as np
from model_code.wealth_and_budget.pensions import (
    calc_experience_for_total_pension_points,
)
from model_code.wealth_and_budget.pensions import calc_total_pension_points


def create_state_space_functions():
    return {
        "get_state_specific_choice_set": state_specific_choice_set,
        "update_continuous_state": get_next_period_experience,
    }


def sparsity_condition(
    period,
    lagged_choice,
    informed,
    job_offer,
    partner_state,
    policy_state,
    education,
    options,
):
    start_age = options["start_age"]
    max_ret_age = options["max_ret_age"]
    min_ret_age_state_space = options["min_ret_age"]

    age = start_age + period
    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age_state_space) & (lagged_choice == 0):
        return False
    # After the maximum retirement age, you must be retired
    elif (age > max_ret_age) & (lagged_choice != 0):
        return False

    elif (age < max_ret_age) and (lagged_choice == 0):
        # If job offer is equal to 0, the state is valid,
        # for every other job offer, the state is proxied to
        # the state where job_offer is 0
        if job_offer == 0:
            return True
        else:
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "education": education,
                "informed": informed,
                "partner_state": partner_state,
                "job_offer": 0,
                "policy_state": policy_state,
            }
            return state_proxy
    elif age > (max_ret_age + 1):
        # If age is larger than max_ret_age + 2, we can also degenerate the policy state to
        # the last policy state (degenerate state) n_policy_states - 1
        if (job_offer == 0) & (policy_state == (options["n_policy_states"] - 1)):
            return True
        else:
            state_proxy = {
                "period": period,
                "lagged_choice": lagged_choice,
                "education": education,
                "informed": informed,
                "partner_state": partner_state,
                "job_offer": 0,
                "policy_state": options["n_policy_states"] - 1,
            }
            return state_proxy
    else:
        return True


def state_specific_choice_set(period, lagged_choice, policy_state, job_offer, options):
    age = period + options["start_age"]
    SRA_pol_state = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(SRA_pol_state, options)

    # retirement is absorbing
    if lagged_choice == 0:
        return np.array([0])
    # Check if the person is not in the voluntary retirement range.
    elif age < min_ret_age_pol_state:
        if job_offer == 0:
            return np.array([1])
        else:
            return np.array([1, 2, 3])
    elif age >= options["max_ret_age"]:
        return np.array([0])
    else:
        if job_offer == 0:
            return np.array([0, 1])
        else:
            return np.array([0, 1, 2, 3])


def apply_retirement_constraint_for_SRA(SRA, options):
    return np.maximum(SRA - options["ret_years_before_SRA"], 63)


def get_next_period_experience(
    period, lagged_choice, policy_state, education, experience, informed, options
):
    """Update experience based on lagged choice and period."""
    max_experience_period = period + options["max_init_experience"]
    exp_years_last_period = (max_experience_period - 1) * experience

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
        period, exp_years_last_period, education, policy_state, informed, options
    )
    # Update if fresh retired
    exp_new_period = jax.lax.select(
        fresh_retired, experience_years_with_penalty, exp_new_period
    )
    return (1 / max_experience_period) * exp_new_period


def calc_experience_years_for_pension_adjustment(
    period, experience_years, education, policy_state, informed, options
):
    """Calculate the reduced experience with early retirement penalty."""
    total_pension_points = calc_total_pension_points(
        education=education,
        experience_years=experience_years,
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
        adjusted_pension_points, education, options
    )
    return reduced_experience_years
