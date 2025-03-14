from model_code.state_space.choice_set import state_specific_choice_set
from model_code.state_space.experience import get_next_period_experience


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
