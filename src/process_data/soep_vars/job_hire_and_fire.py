def generate_job_separation_var(data):
    """This function generates a job separation variable.

    The function creates a new column job_sep which is 1 if the individual got fired
    from the last job. It uses plb0304_h from the soep pl data.

    """
    data.loc[:, "job_sep"] = 0
    data.loc[data["plb0304_h"].isin([1, 3, 5]), "job_sep"] = 1
    return data


def determine_observed_job_offers(data, working_choices, was_fired_last_period):
    """Determine if a job offer is observed and if so what it is. The function
    implements the following rule:

    Assume lagged choice is working (column "lagged_choice" is in working_choices),
    then the state is fully observed:
        - If individual continues working (column "choice" is in working choices):
            - There exists a job offer, i.e. equal to 1
        - If individual does not continue working (column "choice" is not in working choices):
            - Individual got fired then job offer equal 0
            - Individual was not fired then job offer equal to 1

    Assume lagged choice not working (column "lagged_choice" not in working_choices),
    then the state is partially observed:
        - If choice is in working_choices, then the state is fully observed and there is a job offer
        - If choice is not in working choices, then one is not observed

    Lagged choice equal to 0 (retired), will be dropped as only choice equal to 0 is allowed

    Therefore the unobserved job offer states are, where individuals are unemployed and remain unemployed or retire.
    We mark unobsorved states by a state value of -99
    """
    working_this_period = data["choice"].isin(working_choices)
    was_working_last_period = data["lagged_choice"].isin(working_choices)

    data["job_offer"] = -99

    # Individuals working have job offer equal to 1 and are fully observed
    data.loc[working_this_period, "job_offer"] = 1

    # Individuals who are unemployed or retired and are fired this period have job offer
    # equal to 0. This includes individuals with lagged choice unemployment, as they
    # might be interviewed after firing.
    maskfired = (~working_this_period) & was_fired_last_period & was_working_last_period
    data.loc[maskfired, "job_offer"] = 0

    # Everybody who was not fired is also fully observed an has an job offer
    mask_not_fired = (
        (~working_this_period) & (~was_fired_last_period) & was_working_last_period
    )
    data.loc[mask_not_fired, "job_offer"] = 1
    return data
