import pickle

import numpy as np
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.pre_processing.setup_model import setup_and_save_model
from dcegm.solve import get_solve_func_for_model
from model_code.state_space import create_state_space_functions
from model_code.state_space import sparsity_condition
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from model_code.stochastic_processes.partner_transitions import partner_transition
from model_code.utility_functions import create_final_period_utility_functions
from model_code.utility_functions import create_utility_functions
from model_code.wealth_and_budget.budget_equation import budget_constraint
from model_code.wealth_and_budget.savings_grid import create_savings_grid
from specs.derive_specs import generate_derived_and_data_derived_specs


def specify_model(
    path_dict,
    update_spec_for_policy_state,
    policy_state_trans_func,
    params,
    load_model=False,
):
    """Generate model and options dictionaries."""
    # Generate model_specs
    specs, params = generate_derived_and_data_derived_specs(path_dict, params)
    # Assign income shock scale to start_params_all
    params["sigma"] = specs["income_shock_scale"]
    params["interest_rate"] = specs["interest_rate"]
    params["beta"] = specs["discount_factor"]

    # Execute load first step estimation data
    specs = update_spec_for_policy_state(
        specs=specs,
        path_dict=path_dict,
    )

    # Load specifications
    n_periods = specs["n_periods"]
    n_possible_ret_ages = specs["n_possible_ret_ages"]
    n_policy_states = specs["n_policy_states"]
    choices = np.arange(specs["n_choices"], dtype=int)

    options = {
        "state_space": {
            "n_periods": n_periods,
            "choices": choices,
            "endogenous_states": {
                "experience": np.arange(n_periods, dtype=int),
                "education": np.arange(specs["n_education_types"], dtype=int),
                "retirement_age_id": np.arange(n_possible_ret_ages, dtype=int),
                "sparsity_condition": sparsity_condition,
            },
            "exogenous_processes": {
                "policy_state": {
                    "transition": policy_state_trans_func,
                    "states": np.arange(n_policy_states, dtype=int),
                },
                "job_offer": {
                    "transition": job_offer_process_transition,
                    "states": np.arange(2, dtype=int),
                },
                "partner_state": {
                    "transition": partner_transition,
                    "states": np.arange(specs["n_partner_states"], dtype=int),
                },
            },
        },
        "model_params": specs,
    }

    savings_grid = create_savings_grid()

    if load_model:
        model = load_and_setup_model(
            options=options,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            path=path_dict["intermediate_data"] + "model.pkl",
        )

    else:
        model = setup_and_save_model(
            options=options,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            exog_savings_grid=savings_grid,
            path=path_dict["intermediate_data"] + "model.pkl",
        )

    print("Model specified.")
    return model, params


def specify_and_solve_model(
    path_dict,
    file_append,
    params,
    update_spec_for_policy_state,
    policy_state_trans_func,
    load_model,
    load_solution,
):
    """Specify and solve model.

    Also includes possibility to save solutions.

    """

    # Generate model_specs
    model, params = specify_model(
        path_dict=path_dict,
        update_spec_for_policy_state=update_spec_for_policy_state,
        policy_state_trans_func=policy_state_trans_func,
        params=params,
        load_model=load_model,
    )

    solution_file = path_dict["intermediate_data"] + (
        f"solved_models/model_solution" f"_{file_append}.pkl"
    )
    if load_solution:
        solution = pickle.load(open(solution_file, "rb"))
        return solution, model, params
    else:
        solution = {}
        (
            solution["value"],
            solution["policy"],
            solution["endog_grid"],
        ) = get_solve_func_for_model(model)(params)
        pickle.dump(solution, open(solution_file, "wb"))
        return solution, model, params
