import numpy as np
from setup_model.state_space import sparsity_condition

def set_params_and_options():
    start_age = 25
    end_age = 75
    n_periods = end_age - start_age + 1
    resolution_age = 60
    max_retirement_age = 72
    minimum_SRA = 67
    # you can retire four years before minimum_SRA
    min_retirement_age = minimum_SRA - 4
    # you can retire from min retirement age until max retirement age
    n_possible_retirement_ages = max_retirement_age - min_retirement_age + 1
    # when you are (start_age) years old, there can be as many policy states as there are years until (resolution_age)
    n_possible_policy_states = resolution_age - start_age + 1
    # choices: 0 = unemployment, , 1 = work, 2 = retire
    choices = np.array([0, 1, 2])

    options_test = {
        "state_space": {
            "n_periods": n_periods,
            "choices": np.array([0, 1, 2]),
            "endogenous_states": {
                "experience": np.arange(n_periods, dtype=int),
                "policy_state": np.arange(n_possible_policy_states, dtype=int),
                "retirement_age_id": np.arange(n_possible_retirement_ages, dtype=int),
                "sparsity_condition": sparsity_condition,
            },
        },
        "model_params": {
            # info from state spoace used in functions
            "n_periods": n_periods,
            "n_possible_policy_states": n_possible_policy_states,
            # mandatory keywords
            "quadrature_points_stochastic": 5,
            # custom: model structure
            "start_age": start_age,
            "resolution_age": resolution_age,
            # custom: policy environment
            "minimum_SRA": minimum_SRA,
            "max_retirement_age": max_retirement_age,
            "min_retirement_age": min_retirement_age,
            "unemployment_benefits": 5,
            "pension_point_value": 0.3,
            "early_retirement_penalty": 0.036,
            # custom: params estimated outside model
            "belief_update_increment": 0.05,
            "gamma_0": 10,
            "gamma_1": 1,
            "gamma_2": -0.1,
        },
    }

    params_dict_test = {
        "mu": 0.5,  # Risk aversion
        "delta": 4,  # Disutility of work
        "interest_rate": 0.03,
        "lambda": 1e-16,  # Taste shock scale/variance. Almost equal zero = no taste shocks
        "beta": 0.95,  # Discount factor
        "sigma": 1,  # Income shock scale/variance.
    }
    return params_dict_test, options_test