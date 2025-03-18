import numpy as np


def state_specific_choice_set(
    period, lagged_choice, sex, policy_state, job_offer, health, options
):
    """This function is called in the model generation. Therefore we do not need to care
    if it is implemented efficiently. Rather we want to make the model restriction on choices
    explicit.
    """
    age = period + options["start_age"]
    SRA_pol_state = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    ret_age_long_insured = retirement_age_long_insured(SRA_pol_state, options)

    # If somebody is death, we assign a dummy choice set of [0]
    if health == options["death_health_var"]:
        return np.array([0])
    # retirement is absorbing
    elif lagged_choice == 0:
        return np.array([0])
    # Above the maximum retirement age, also everybody needs to be retired
    elif age >= options["max_ret_age"]:
        return np.array([0])
    else:
        # For the rest of the life span, we now check a few rules and delete choices
        # accordingly.
        # Start with a full choice set:
        choice_set = np.array([0, 1, 2, 3])
        # You can only choose retirement if the age is above the age of retirement for
        # long insured people. Or if you are disabled. Check first if you are able to retire
        # otherwise delete the retirement choice.
        able_to_retire = (age >= ret_age_long_insured) | (
            health == options["disabled_health_var"]
        )
        if not able_to_retire:
            choice_set = choice_set[choice_set != 0]

        # If you are above the SRA, you can't decide on unemployment anymore.
        # Note, then you definitely are able to retire, so retirement is in the choice set.
        if age >= SRA_pol_state:
            choice_set = choice_set[choice_set != 1]

        # If you don't have a job offer, you can't decide on part-time or full-time work.
        if job_offer == 0:
            choice_set = choice_set[choice_set != 2]
            choice_set = choice_set[choice_set != 3]
        # Men can't work part-time. We delete choice 2 for them.
        if sex == 0:
            choice_set = choice_set[choice_set != 2]
        return choice_set


def retirement_age_long_insured(SRA, options):
    """Everyone can retire 4 years before the SRA but must be at least at 63 y/o.
    That means that we assume 1) everyone qualifies for "Rente für langjährig Versicherte" and 2) that
    the threshhold for "Rente für langjährig Versicherte" moves with the SRA. "Rente für besonders langjährig
    Versicherte" only differs with respect to deductions. Not with respect to entry age. We introduce the
    lower bound of 63 as this is the current law, even for individuals with SRA below 67.
    """
    return np.maximum(SRA - options["ret_years_before_SRA"], 63)
