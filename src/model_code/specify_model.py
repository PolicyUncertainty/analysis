import pickle

import jax.numpy as jnp
import numpy as np
from dcegm.pre_processing.setup_model import load_and_setup_model
from dcegm.pre_processing.setup_model import setup_and_save_model
from dcegm.solve import get_solve_func_for_model
from model_code.policy_processes.informed_state_transition import (
    informed_transition,
)
from model_code.policy_processes.select_policy_belief import (
    select_expectation_functions_and_model_sol_names,
)
from model_code.state_space import create_state_space_functions
from model_code.stochastic_processes.health_transition import health_transition
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from model_code.stochastic_processes.partner_transitions import partner_transition
from model_code.utility.bequest_utility import create_final_period_utility_functions
from model_code.utility.utility_functions import create_utility_functions
from model_code.wealth_and_budget.budget_equation import budget_constraint
from model_code.wealth_and_budget.savings_grid import create_savings_grid
from set_paths import get_model_resutls_path
from specs.derive_specs import generate_derived_and_data_derived_specs


def specify_model(
    path_dict,
    update_spec_for_policy_state,
    policy_state_trans_func,
    params,
    load_model=False,
    model_type="solution",
):
    """Generate model and options dictionaries."""
    # Generate model_specs
    specs = generate_derived_and_data_derived_specs(path_dict)

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
    n_policy_states = specs["n_policy_states"]
    choices = np.arange(specs["n_choices"], dtype=int)

    # Create savings grid
    savings_grid = create_savings_grid()

    # Experience grid
    experience_grid = jnp.linspace(0, 1, specs["n_experience_grid_points"])

    options = {
        "state_space": {
            "min_period_batch_segments": [33, 44],
            "n_periods": n_periods,
            "choices": choices,
            "endogenous_states": {
                "education": np.arange(specs["n_education_types"], dtype=int),
                "sex": np.arange(specs["n_sexes"], dtype=int),
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
                "health": {
                    "transition": health_transition,
                    "states": np.arange(specs["n_health_states"], dtype=int),
                },
            },
            "continuous_states": {
                "wealth": savings_grid,
                "experience": experience_grid,
            },
        },
        "model_params": specs,
    }
    informed_states = np.arange(2, dtype=int)
    if model_type == "solution":
        # Set informed state as not changing state
        options["state_space"]["endogenous_states"]["informed"] = informed_states
        # Determine path
        model_path = path_dict["intermediate_data"] + "model_spec_solution.pkl"
        sim_model = False

    elif model_type == "simulation":
        # Set informed state as exogenous changing state
        options["state_space"]["exogenous_processes"]["informed"] = {
            "transition": informed_transition,
            "states": informed_states,
        }
        # Determine path
        model_path = path_dict["intermediate_data"] + "model_spec_simulation.pkl"
        sim_model = True
    else:
        raise ValueError("model_type must be either 'solution' or 'simulation'")

    if load_model:
        model = load_and_setup_model(
            options=options,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            # shock_functions=shock_function_dict(),
            path=model_path,
            sim_model=sim_model,
        )

    else:
        model = setup_and_save_model(
            options=options,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            # shock_functions=shock_function_dict(),
            path=model_path,
            sim_model=sim_model,
        )

    print("Model specified.")
    return model, params


def specify_and_solve_model(
    path_dict,
    file_append,
    params,
    expected_alpha,
    resolution,
    load_model,
    load_solution,
):
    """Specify and solve model.

    Also includes possibility to save solutions.

    """
    (
        update_funcs,
        transition_funcs,
        model_sol_names,
    ) = select_expectation_functions_and_model_sol_names(
        path_dict,
        expected_alpha=expected_alpha,
        sim_alpha=None,
        resolution=resolution,
    )

    # Generate model_specs
    model, params = specify_model(
        path_dict=path_dict,
        update_spec_for_policy_state=update_funcs["solution"],
        policy_state_trans_func=transition_funcs["solution"],
        params=params,
        load_model=load_model,
        model_type="solution",
    )

    # check if folder of model objects exits:
    solve_folder = get_model_resutls_path(path_dict, file_append)
    solution_file = solve_folder["solution"] + model_sol_names["solution"]

    if load_solution is None:
        solution = {}
        (
            solution["value"],
            solution["policy"],
            solution["endog_grid"],
        ) = get_solve_func_for_model(model)(params)
        return solution, model, params
    elif load_solution:
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
