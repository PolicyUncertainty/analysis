import jax.numpy as jnp


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
    SRA_at_resolution = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    actual_retirement_age = options["min_ret_age"] + retirement_age_id

    # calculate applicable SRA and pension deduction/increase factor
    # (malus for early retirement, bonus for late retirement)
    pension_factor = 1 - (actual_retirement_age - SRA_at_resolution) * ERP
    retirement_income = pension_point_value * experience * pension_factor

    means_test = savings_end_of_previous_period < options["unemployment_wealth_thresh"]
    # Unemployment benefits
    unemployment_benefits = means_test * options["unemployment_benefits"]
    # Labor income
    labor_income = (
        gamma_0
        + gamma_1 * experience
        + gamma_2 * experience**2
        + income_shock_previous_period
    )
    labor_income_with_min = jnp.maximum(labor_income, options["min_wage"])

    # bools of last period decision: income is payed in following period!
    was_worker = lagged_choice == 1
    was_retired = lagged_choice == 2

    income = (
        jnp.maximum(
            was_worker * labor_income_with_min + was_retired * retirement_income,
            unemployment_benefits,
        )
        * 12
    )

    # calculate beginning of period wealth M_t
    wealth = (1 + params["interest_rate"]) * savings_end_of_previous_period + income

    return wealth
