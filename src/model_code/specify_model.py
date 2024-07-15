import numpy as np
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.pre_processing.setup_model import setup_and_save_model
from model_code.derive_specs import generate_specs_and_update_params
from model_code.state_space import create_state_space_functions
from model_code.state_space import sparsity_condition
from model_code.utility_functions.final_period import (
    create_final_period_utility_functions,
)
from model_code.utility_functions.main_utility_functions import (
    create_main_utility_functions,
)
from model_code.utility_functions.model_switch_utility_functions import (
    create_switch_utility_functions_dict,
)
from model_code.utility_functions.old_age_utility_functions import (
    create_old_age_utility_functions,
)
from model_code.wealth_and_budget.main_budget_equation import main_budget_constraint
from model_code.wealth_and_budget.old_age_budget_equation import (
    old_age_budget_constraint,
)
from model_code.wealth_and_budget.savings_grid import create_savings_grid


def specify_model(
    path_dict,
    update_spec_for_policy_state,
    policy_state_trans_func,
    params,
    load_model=False,
):
    """Generate model and options dictionaries."""
    # Generate model_specs
    specs, params = generate_specs_and_update_params(path_dict, params)

    # Execute load first step estimation data
    specs = update_spec_for_policy_state(
        specs=specs,
        path_dict=path_dict,
    )

    # Load specifications
    n_periods_main = specs["n_periods_main"]
    n_possible_ret_ages = specs["n_possible_ret_ages"]
    n_policy_states = specs["n_policy_states"]
    choices = np.arange(specs["n_choices"], dtype=int)
    n_experience_levels_max = n_periods_main + specs["max_init_experience"]

    options_main = {
        "state_space": {
            "n_periods": n_periods_main,
            "choices": choices,
            "endogenous_states": {
                "experience": np.arange(n_experience_levels_max, dtype=int),
                "education": np.arange(specs["n_education_types"], dtype=int),
                "retirement_age_id": np.arange(n_possible_ret_ages, dtype=int),
                "sparsity_condition": sparsity_condition,
            },
            "exogenous_processes": {
                "policy_state": {
                    "transition": policy_state_trans_func,
                    "states": np.arange(n_policy_states, dtype=int),
                },
            },
        },
        "model_params": specs,
    }

    options_old_age = {
        "state_space": {
            "n_periods": specs["n_periods_old_age"],
            "endogenous_states": {
                "education": np.arange(specs["n_education_types"], dtype=int),
                "deduction_state": np.arange(specs["n_deduction_states"], dtype=int),
            },
        },
        "model_params": specs,
    }

    options_old_age["state_space"]["endogenous_states"]["experience"] = np.arange(
        specs["exp_cap"] + 1, dtype=int
    )

    savings_grid = create_savings_grid()

    if load_model:
        model_old_age = load_and_setup_model(
            options=options_old_age,
            utility_functions=create_old_age_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=old_age_budget_constraint,
            path=path_dict["intermediate_data"] + "model_old_age.pkl",
        )

        print("Old age model specified.")
        model_main = load_and_setup_model(
            options=options_main,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_main_utility_functions(),
            utility_functions_final_period=create_switch_utility_functions_dict(
                model_old_age
            ),
            budget_constraint=main_budget_constraint,
            path=path_dict["intermediate_data"] + "model_main.pkl",
        )

    else:
        model_old_age = setup_and_save_model(
            options=options_old_age,
            utility_functions=create_old_age_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=old_age_budget_constraint,
            exog_savings_grid=savings_grid,
            path=path_dict["intermediate_data"] + "model_old_age.pkl",
        )

        print("Old age model specified.")
        model_main = setup_and_save_model(
            options=options_main,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_main_utility_functions(),
            utility_functions_final_period=create_switch_utility_functions_dict(
                model_old_age
            ),
            budget_constraint=main_budget_constraint,
            exog_savings_grid=savings_grid,
            path=path_dict["intermediate_data"] + "model_main.pkl",
        )

    print("Model specified.")

    model_collection = {
        "model_main": model_main,
        "model_old_age": model_old_age,
    }
    return model_collection, params
