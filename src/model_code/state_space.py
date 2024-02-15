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
    SRA_pol_state = options["min_SRA"] + retirement_age_id * options["SRA_grid_size"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(SRA_pol_state, options)

    age = start_age + period
    actual_retirement_age = min_ret_age_state_space + retirement_age_id
    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age_pol_state) & (lagged_choice == 2):
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

    age = period + options["start_age"]

    if choice == 0:
        return next_state
    elif choice == 1:
        if experience < options["exp_cap"]:
            next_state["experience"] = experience + 1
            return next_state
        else:
            return next_state
    elif choice == 2:
        if lagged_choice == 2:
            return next_state
        else:
            next_state["retirement_age_id"] = age - options["min_ret_age"]
    else:
        raise ValueError("Choice not recognized in update_state_space")

    return next_state


def state_specific_choice_set(period, lagged_choice, policy_state, options):
    age = period + options["start_age"]
    SRA_pol_state = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(SRA_pol_state, options)

    if age < min_ret_age_pol_state:
        return np.array([0, 1])
    elif age >= options["max_ret_age"]:
        return np.array([2])

    # If all choices possible, then check if already retired
    if lagged_choice == 2:  # retirement is absorbing
        return np.array([2])
    else:
        return np.array([0, 1, 2])


def apply_retirement_constraint_for_SRA(SRA, options):
    return np.maximum(SRA - options["ret_years_before_SRA"], 63)
