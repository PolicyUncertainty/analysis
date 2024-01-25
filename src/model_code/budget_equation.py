def budget_constraint(
    lagged_choice,  # d_{t-1}
    experience,
    policy_state,  # current applicable SRA identifyer
    retirement_age_id,
    savings_end_of_previous_period,  # A_{t-1}
    income_shock_previous_period,  # epsilon_{t - 1}
    params,
    options,
):
    # fetch necessary parameters (gammas for wage, pension_point_value & ERP for pension)
    gamma_0 = options["gamma_0"]
    gamma_1 = options["gamma_1"]
    gamma_2 = options["gamma_2"]
    pension_point_value = options["pension_point_value"]
    ERP = options["early_retirement_penalty"]

    # generate actual retirement age and SRA at resolution
    SRA_at_resolution = (
        options["minimum_SRA"] + policy_state * options["belief_update_increment"]
    )
    actual_retirement_age = options["min_retirement_age"] + retirement_age_id

    # calculate applicable SRA and pension deduction/increase factor
    # (malus for early retirement, bonus for late retirement)

    pension_factor = 1 - (actual_retirement_age - SRA_at_resolution) * ERP

    # decision bools
    is_unemployed = lagged_choice == 0
    is_worker = lagged_choice == 1
    is_retired = lagged_choice == 2

    # decision-specific income
    unemployment_benefits = options["unemployment_benefits"]
    labor_income = (
        gamma_0
        + gamma_1 * experience
        + gamma_2 * experience**2
        + income_shock_previous_period
    )
    retirement_income = pension_point_value * experience * pension_factor

    income = (
        is_unemployed * unemployment_benefits
        + is_worker * labor_income
        + is_retired * retirement_income
    )

    # calculate beginning of period wealth M_t
    wealth = (1 + params["interest_rate"]) * savings_end_of_previous_period + income

    return wealth