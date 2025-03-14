import numpy as np


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
    """Everyone can retire 4 years before the SRA but must be at least at 63 y/o.
    That means that we assume 1) everyone qualifies for "Rente für langjährig Versicherte" and 2) that
    the threshhold for "Rente für langjährig Versicherte" moves with the SRA.
    """
    return np.maximum(SRA - options["ret_years_before_SRA"], 63)
