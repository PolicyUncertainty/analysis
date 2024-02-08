import numpy as np
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.pre_processing.setup_model import setup_and_save_model
from model_code.budget_equation import budget_constraint
from model_code.state_space import create_state_space_functions
from model_code.state_space import sparsity_condition
from model_code.utility_functions import create_final_period_utility_functions
from model_code.utility_functions import create_utility_functions


def specify_model(project_specs, load_model=False):
    # Load specifications
    n_periods = project_specs["n_periods"]
    n_possible_ret_ages = project_specs["n_possible_ret_ages"]
    n_possible_policy_states = project_specs["n_possible_policy_states"]
    choices = np.arange(project_specs["n_choices"], dtype=int)

    options = {
        "state_space": {
            "n_periods": n_periods,
            "choices": choices,
            "endogenous_states": {
                "experience": np.arange(n_periods, dtype=int),
                "policy_state": np.arange(n_possible_policy_states, dtype=int),
                "retirement_age_id": np.arange(n_possible_ret_ages, dtype=int),
                "sparsity_condition": sparsity_condition,
            },
        },
        "model_params": project_specs,
    }

    if load_model:
        model = load_and_setup_model(
            options=options,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            path="model.pkl",
        )

    else:
        model = setup_and_save_model(
            options=options,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            path="model.pkl",
        )

    return model, options
