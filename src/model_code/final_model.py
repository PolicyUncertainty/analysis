def solve_final_period_scalar(
    choice,
    begin_of_period_resources,
    params,
    options,
    compute_utility,
    compute_marginal_utility,
):
    """Compute optimal consumption policy and value function in the final period.

    In the last period, everything is consumed, i.e. consumption = savings.

    Args:
        state (np.ndarray): 1d array of shape (n_state_variables,) containing the
            period-specific state vector.
        choice (int): The agent's choice in the current period.
        begin_of_period_resources (float): The agent's begin of period resources.
        compute_utility (callable): Function for computation of agent's utility.
        compute_marginal_utility (callable): Function for computation of agent's
        params (dict): Dictionary of model parameters.
        options (dict): Options dictionary.

    Returns:
        tuple:

        - consumption (float): The agent's consumption in the final period.
        - value (float): The agent's value in the final period.
        - marginal_utility (float): The agent's marginal utility .

    """

    # eat everything
    consumption = begin_of_period_resources

    # utility & marginal utility of eating everything
    value = compute_utility(
        consumption=begin_of_period_resources, choice=choice, params=params
    )

    marginal_utility = compute_marginal_utility(
        consumption=begin_of_period_resources, params=params
    )

    return marginal_utility, value, consumption