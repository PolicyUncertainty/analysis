import numpy as np


def create_state_space_functions():
    return {
        "get_next_period_state": update_state_space,
        "get_state_specific_choice_set": state_specific_choice_set,
    }


def sparsity_condition(period, lagged_choice, retirement_age_id, experience, options):
    start_age = options["start_age"]
    max_ret_age = options["max_ret_age"]
    max_init_experience = options["max_init_experience"]
    min_ret_age_state_space = options["min_ret_age"]

    age = start_age + period
    actual_retirement_age = min_ret_age_state_space + retirement_age_id
    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age_state_space) & (lagged_choice == 2):
        return False
    # After the maximum retirement age, you must be retired
    elif (age > max_ret_age) & (lagged_choice != 2):
        return False
    # If you weren't retired last period, your actual retirement age is kept at minimum
    elif (lagged_choice != 2) & (retirement_age_id > 0):
        return False
    # If you are retired, your actual retirement age can at most be your current age
    elif (lagged_choice == 2) & (age <= actual_retirement_age):
        return False
    # If you have not worked last period, you can't have worked all your live
    elif (
        (lagged_choice != 1)
        & (period + max_init_experience == experience)
        & (period > 0)
    ):
        return False
    # You cannot have more experience than your age
    elif experience > period + max_init_experience:
        return False
    elif experience > options["exp_cap"]:
        return False
    else:
        return True


def update_state_space(
    period, choice, lagged_choice, retirement_age_id, experience, options
):
    next_state = dict()

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice

    # Create work bools
    unemployment_bool = choice == 0
    work_bool = choice == 1
    retirement_bool = choice == 2

    # Update experience
    below_exp_cap_bool = experience < options["exp_cap"]
    exp_update_bool = below_exp_cap_bool * work_bool
    next_state["experience"] = experience + exp_update_bool

    # Update retirement age. First create possible retirement id and then check if
    # retirement is chosen and not already retired
    poss_ret_id = period + options["start_age"] - options["min_ret_age"]
    not_retired_bool = lagged_choice != 2
    ret_age_update_bool = not_retired_bool * retirement_bool
    next_state["retirement_age_id"] = (
        ret_age_update_bool * poss_ret_id
        + (1 - ret_age_update_bool) * retirement_age_id
    )

    return next_state


def state_specific_choice_set(period, lagged_choice, policy_state, options):
    age = period + options["start_age"]
    SRA_pol_state = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(SRA_pol_state, options)

    # retirement is absorbing
    if lagged_choice == 2:
        return np.array([2])
    # Check if the person is not in the voluntary retirement range.
    elif age < min_ret_age_pol_state:
        return np.array([0, 1])
    elif age >= options["max_ret_age"]:
        return np.array([2])
    else:
        return np.array([0, 1, 2])


def apply_retirement_constraint_for_SRA(SRA, options):
    return np.maximum(SRA - options["ret_years_before_SRA"], 63)
