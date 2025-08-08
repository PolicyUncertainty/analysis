import pickle
from copy import deepcopy

import dcegm
import jax.numpy as jnp
import numpy as np

from model_code.policy_processes.informed_state_transition import (
    informed_transition,
)
from model_code.policy_processes.select_policy_belief import (
    select_sim_policy_function_and_update_specs,
    select_solution_transition_func_and_update_specs,
)
from model_code.state_space.state_space import create_state_space_functions
from model_code.stochastic_processes.health_transition import health_transition
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from model_code.stochastic_processes.partner_transitions import partner_transition
from model_code.taste_shocks import shock_function_dict
from model_code.utility.bequest_utility import create_final_period_utility_functions
from model_code.utility.utility_functions_add import create_utility_functions
from model_code.wealth_and_budget.assets_grid import create_end_of_period_assets
from model_code.wealth_and_budget.budget_equation import budget_constraint
from set_paths import get_model_resutls_path
from specs.derive_specs import generate_derived_and_data_derived_specs


def specify_simple_model(
    path_dict,
    specs,
    load_model=False,
):
    """Generate model class."""

    SRA_belief_solution, specs = select_solution_transition_func_and_update_specs(
        path_dict=path_dict,
        specs=specs,
        subj_unc=False,
        custom_resolution_age=None,
    )

    # Create savings grid
    savings_grid = create_end_of_period_assets()

    # Experience grid
    experience_grid = jnp.linspace(0, 1, specs["n_experience_grid_points"])

    model_config = {
        "min_period_batch_segments": [33, 44],
        "n_periods": specs["n_periods"],
        "choices": np.arange(specs["n_choices"], dtype=int),
        "deterministic_states": {
            "education": [0],
            "sex": [0],
            "partner_state": [0],
            "health": [0],
            "informed": [0, 1],
        },
        "stochastic_states": {
            "policy_state": np.arange(specs["n_policy_states"], dtype=int),
            "job_offer": np.arange(2, dtype=int),
        },
        "continuous_states": {
            "assets_end_of_period": savings_grid / specs["wealth_unit"],
            "experience": experience_grid,
        },
        "n_quad_points": specs["n_quad_points"],
    }
    stochastic_states_transitions = {
        "policy_state": SRA_belief_solution,
        "job_offer": job_offer_process_transition,
        # "partner_state": partner_transition,
        # "health": health_transition,
    }

    # # Now we use the alternative sim specification to define informed in the solution
    # # as deterministic state (type) and in the simulation as stochastic state.
    # informed_states = np.arange(2, dtype=int)
    # model_config_sim = deepcopy(model_config)
    # stochastic_states_transitions_sim = deepcopy(stochastic_states_transitions)
    #
    # # First add it as a deterministic state
    # model_config["deterministic_states"]["informed"] = informed_states

    model_path = path_dict["intermediate_data"] + "model_simple.pkl"

    if load_model:
        model = dcegm.setup_model(
            model_specs=specs,
            model_config=model_config,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            shock_functions=shock_function_dict(),
            stochastic_states_transitions=stochastic_states_transitions,
            model_load_path=model_path,
        )

    else:
        model = dcegm.setup_model(
            model_specs=specs,
            model_config=model_config,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            shock_functions=shock_function_dict(),
            stochastic_states_transitions=stochastic_states_transitions,
            model_save_path=model_path,
        )

    print("Model specified.")
    return model


def specify_and_solve_simple_model(
    path_dict,
    file_append,
    params,
    load_model,
    load_solution,
):
    """Specify and solve model.

    Also includes possibility to save solutions.

    """

    specs = generate_derived_and_data_derived_specs(path_dict)

    # Generate model_specs
    model = specify_simple_model(
        path_dict=path_dict,
        specs=specs,
        load_model=load_model,
    )

    # check if folder of model objects exits:
    solve_folder = get_model_resutls_path(path_dict, file_append)
    sol_name = f"sol_simple.pkl"

    solution_file = solve_folder["solution"] + sol_name

    if load_solution is None:
        model_solved = model.solve(params)
        return model_solved
    elif load_solution:
        model_solved = model.solve(params, load_sol_path=solution_file)
        return model_solved
    else:
        model_solved = model.solve(params, save_sol_path=solution_file)
        return model_solved
