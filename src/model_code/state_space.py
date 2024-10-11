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


def sparsity_condition(period, lagged_choice, options):
    start_age = options["start_age"]
    max_ret_age = options["max_ret_age"]
    min_ret_age_state_space = options["min_ret_age"]

    age = start_age + period
    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age_state_space) & (lagged_choice == 2):
        return False
    # After the maximum retirement age, you must be retired
    elif (age > max_ret_age) & (lagged_choice != 2):
        return False
    else:
        return True


def state_specific_choice_set(period, lagged_choice, policy_state, job_offer, options):
    age = period + options["start_age"]
    SRA_pol_state = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(SRA_pol_state, options)

    # retirement is absorbing
    if lagged_choice == 2:
        return np.array([2])
    # Check if the person is not in the voluntary retirement range.
    elif age < min_ret_age_pol_state:
        if job_offer == 0:
            return np.array([0])
        else:
            return np.array([0, 1])
    elif age >= options["max_ret_age"]:
        return np.array([2])
    else:
        if job_offer == 0:
            return np.array([0, 2])
        else:
            return np.array([0, 1, 2])


def apply_retirement_constraint_for_SRA(SRA, options):
    return np.maximum(SRA - options["ret_years_before_SRA"], 63)


def get_next_period_experience(
    period, lagged_choice, policy_state, education, experience, options
):
    """Update experience based on lagged choice and period."""
    max_experience_period = period + options["max_init_experience"]
    exp_years_last_period = (max_experience_period - 1) * experience

    # Update if working
    exp_new_period = exp_years_last_period + (lagged_choice == 1)

    # If retired, then we update experience according to the deduction function
    degenerate_state_id = options["n_policy_states"] - 1
    fresh_retired = (degenerate_state_id != policy_state) & (lagged_choice == 2)
    experience_years_with_penalty = calc_experience_years_for_pension_adjustment(
        period, exp_years_last_period, education, policy_state, options
    )
    # Update if fresh retired
    exp_new_period = (
        1 - fresh_retired
    ) * exp_new_period + fresh_retired * experience_years_with_penalty
    return (1 / max_experience_period) * exp_new_period


def calc_experience_years_for_pension_adjustment(
    period, experience_years, education, policy_state, options
):
    """Calculate the reduced experience with early retirement penalty."""
    total_pension_points = calc_total_pension_points(
        education=education,
        experience_years=experience_years,
        options=options,
    )
    # retirement age is last periods age
    actual_retirement_age = options["start_age"] + period - 1
    # SRA at retirement
    SRA_at_retirement = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    # deduction (bonus) factor for early (late) retirement
    ERP = options["early_retirement_penalty"]
    pension_deduction = (SRA_at_retirement - actual_retirement_age) * ERP
    pension_factor = 1 - pension_deduction
    reduced_pension_points = pension_factor * total_pension_points

    reduced_experience_years = calc_experience_for_total_pension_points(
        reduced_pension_points, education, options
    )
    return reduced_experience_years
