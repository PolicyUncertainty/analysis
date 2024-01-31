import numpy as np


def create_state_space_functions():
    return {
        "get_next_period_state": update_state_space,
        "get_state_specific_choice_set": state_specific_choice_set,
    }


def sparsity_condition(
    period, lagged_choice, policy_state, retirement_age_id, experience, options
):
    min_ret_age = options["min_ret_age"]
    start_age = options["start_age"]
    max_ret_age = options["max_ret_age"]
    n_policy_states = options["n_possible_policy_states"]
    max_init_experience = options["max_init_experience"]

    age = start_age + period
    actual_retirement_age = min_ret_age + retirement_age_id
    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age) & (lagged_choice == 2):
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
    # Starting from resolution age, there is no more adding of policy states.
    elif policy_state > n_policy_states - 1:
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
    # # The policy state we need to consider increases by one increment
    # # per period.
    # elif policy_state > period:
    #     return False
    else:
        return True


def update_state_space(
    period, choice, lagged_choice, policy_state, retirement_age_id, experience, options
):
    next_state = dict()

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice

    age = period + options["start_age"]

    next_state["policy_state"] = policy_state

    if lagged_choice == 2:  # Retirement
        next_state["retirement_age_id"] = retirement_age_id
    elif choice == 2:  # Retirement
        next_state["retirement_age_id"] = age - options["min_ret_age"]
    elif (choice == 1) & (experience < options["exp_cap"]):  # Work
        next_state["experience"] = experience + 1

    return next_state


def state_specific_choice_set(period, lagged_choice, policy_state, options):
    age = period + options["start_age"]
    min_individual_retirement_age = (
        options["min_ret_age"] + policy_state * options["belief_update_increment"]
    )

    if age < min_individual_retirement_age:
        return np.array([0, 1])
    elif age >= options["max_ret_age"]:
        return np.array([2])
    elif lagged_choice == 2:  # retirement is absorbing
        return np.array([2])
    else:
        return np.array([0, 1, 2])
