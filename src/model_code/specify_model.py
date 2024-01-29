import numpy as np
from dcegm.pre_processing.setup_model import setup_model
from model_code.budget_equation import budget_constraint
from model_code.state_space import create_state_space_functions
from model_code.state_space import sparsity_condition
from model_code.utility_functions import create_final_period_utility_functions
from model_code.utility_functions import create_utility_functions


def specify_model(project_specs):
    # Load specifications
    n_possible_ret_ages = project_specs["n_possible_ret_ages"]

    # when you are (start_age) years old, there can be as many policy states as there are years until (resolution_age)
    n_possible_policy_states = resolution_age - start_age + 1
    choices = np.array([0, 1, 2])

    options = {
        "state_space": {
            "n_periods": n_periods,
            "choices": np.array([0, 1, 2]),
            "endogenous_states": {
                "experience": np.arange(n_periods, dtype=int),
                "policy_state": np.arange(n_possible_policy_states, dtype=int),
                "retirement_age_id": np.arange(n_possible_ret_ages, dtype=int),
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

    params = {
        "mu": 0.5,  # Risk aversion
        "delta": 4,  # Disutility of work
        "interest_rate": 0.03,
        "lambda": 1e-16,  # Taste shock scale/variance. Almost equal zero = no taste shocks
        "beta": 0.95,  # Discount factor
        "sigma": 1,  # Income shock scale/variance.
    }
    model = setup_model(
        options=options,
        state_space_functions=create_state_space_functions(),
        utility_functions=create_utility_functions(),
        utility_functions_final_period=create_final_period_utility_functions(),
        budget_constraint=budget_constraint,
    )
    return model, params, options
