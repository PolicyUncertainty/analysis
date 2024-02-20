import numpy as np
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.pre_processing.setup_model import setup_and_save_model
from model_code.belief_process import expected_SRA_probs_estimation
from model_code.budget_equation import budget_constraint
from model_code.state_space import create_state_space_functions
from model_code.state_space import sparsity_condition
from model_code.utility_functions import create_final_period_utility_functions
from model_code.utility_functions import create_utility_functions


def specify_model(project_specs, model_data_path, load_model=False, step="estimation"):
    # Load specifications
    n_periods = project_specs["n_periods"]
    n_possible_ret_ages = project_specs["n_possible_ret_ages"]
    n_policy_states = project_specs["n_policy_states"]
    choices = np.arange(project_specs["n_choices"], dtype=int)

    if step == "estimation":
        exog_trans_func = expected_SRA_probs_estimation

    options = {
        "state_space": {
            "n_periods": n_periods,
            "choices": choices,
            "endogenous_states": {
                "experience": np.arange(n_periods, dtype=int),
                "retirement_age_id": np.arange(n_possible_ret_ages, dtype=int),
                "sparsity_condition": sparsity_condition,
            },
            "exogenous_processes": {
                "policy_state": {
                    "transition": expected_SRA_probs_estimation,
                    "states": np.arange(n_policy_states, dtype=int),
                },
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
            path=model_data_path + "model.pkl",
        )

    else:
        model = setup_and_save_model(
            options=options,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            path=model_data_path + "model.pkl",
        )

    return model, options
