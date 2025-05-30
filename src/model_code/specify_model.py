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
from model_code.utility.utility_functions import create_utility_functions
from model_code.wealth_and_budget.assets_grid import create_end_of_period_assets
from model_code.wealth_and_budget.budget_equation import budget_constraint
from set_paths import get_model_resutls_path
from specs.derive_specs import generate_derived_and_data_derived_specs


def specify_model(
    path_dict,
    subj_unc,
    custom_resolution_age,
    sim_specs=None,
    load_model=False,
):
    """Generate model class."""

    # Generate model_specs
    specs = generate_derived_and_data_derived_specs(path_dict)

    SRA_belief_solution, specs = select_solution_transition_func_and_update_specs(
        path_dict=path_dict,
        specs=specs,
        subj_unc=subj_unc,
        custom_resolution_age=custom_resolution_age,
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
            "education": np.arange(specs["n_education_types"], dtype=int),
            "sex": np.arange(specs["n_sexes"], dtype=int),
        },
        "stochastic_states": {
            "policy_state": np.arange(specs["n_policy_states"], dtype=int),
            "job_offer": np.arange(2, dtype=int),
            "partner_state": np.arange(specs["n_partner_states"], dtype=int),
            "health": np.arange(specs["n_all_health_states"], dtype=int),
        },
        "continuous_states": {
            "assets_end_of_period": savings_grid,
            "experience": experience_grid,
        },
        "n_quad_points": specs["n_quad_points"],
    }
    stochastic_states_transitions = {
        "policy_state": SRA_belief_solution,
        "job_offer": job_offer_process_transition,
        "partner_state": partner_transition,
        "health": health_transition,
    }

    # Now we use the alternative sim specification to define informed in the solution
    # as deterministic state (type) and in the simulation as stochastic state.
    informed_states = np.arange(2, dtype=int)
    model_config_sim = deepcopy(model_config)
    stochastic_states_transitions_sim = deepcopy(stochastic_states_transitions)

    # First add it as a deterministic state
    model_config["deterministic_states"]["informed"] = informed_states

    if sim_specs is not None:
        # Now as stochastic in the sim objects
        model_config_sim["stochastic_states"]["informed"] = informed_states
        stochastic_states_transitions_sim["informed"] = informed_transition

        transition_func_sim, specs = select_sim_policy_function_and_update_specs(
            specs=specs,
            subj_unc=subj_unc,
            announcement_age=sim_specs["announcement_age"],
            SRA_at_start=sim_specs["SRA_at_start"],
            SRA_at_retirement=sim_specs["SRA_at_retirement"],
        )
        stochastic_states_transitions_sim["policy_state"] = transition_func_sim

        # Now specify the dict:
        alternative_sim_specifications = {
            "model_config": model_config_sim,
            "stochastic_states_transitions": stochastic_states_transitions_sim,
            "state_space_functions": create_state_space_functions(),
            "budget_constraint": budget_constraint,
            "shock_functions": shock_function_dict(),
        }

    else:
        alternative_sim_specifications = None

    model_path = path_dict["intermediate_data"] + "model.pkl"

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
            alternative_sim_specifications=alternative_sim_specifications,
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
            alternative_sim_specifications=alternative_sim_specifications,
        )

    print("Model specified.")
    return model


def specify_and_solve_model(
    path_dict,
    file_append,
    params,
    subj_unc,
    custom_resolution_age,
    load_model,
    load_solution,
    sim_specs=None,
):
    """Specify and solve model.

    Also includes possibility to save solutions.

    """

    # Generate model_specs
    model = specify_model(
        path_dict=path_dict,
        subj_unc=subj_unc,
        custom_resolution_age=custom_resolution_age,
        load_model=load_model,
        sim_specs=sim_specs,
    )

    # check if folder of model objects exits:
    solve_folder = get_model_resutls_path(path_dict, file_append)

    # Generate name of solution
    if subj_unc:
        resolution_age = model.model_specs["resolution_age"]
        sol_name = f"sol_subj_unc_{resolution_age}.pkl"
    else:
        sol_name = "sol_no_subj_unc.pkl"

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
