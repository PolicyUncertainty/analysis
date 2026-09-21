import pickle
from copy import deepcopy

import dcegm
import jax
import jax.numpy as jnp
import numpy as np

from model_code.policy_processes.informed_state_transition import (
    informed_transition,
)
from model_code.policy_processes.select_policy_belief import (
    select_sim_policy_function_and_update_specs,
    select_solution_transition_func_and_update_specs,
)
from model_code.state_space.experience import experience_grid_from_state
from model_code.state_space.state_space import create_state_space_functions
from model_code.stochastic_processes.health_transition import health_transition
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from model_code.stochastic_processes.partner_transitions import partner_transition
from model_code.taste_shocks import shock_function_dict
from model_code.utility.bequest_utility import create_final_period_utility_functions
from model_code.wealth_and_budget.assets_grid import create_end_of_period_assets
from model_code.wealth_and_budget.budget_equation import budget_constraint
from set_paths import get_model_results_path
from specs.derive_specs import generate_derived_and_data_derived_specs


def create_model_config_wo_informed(
    specs,
    sex_type,
    edu_type,
    upper_envelope_method=None,
    income_shock_batch_size=None,
):
    """Build the model config.

    ``upper_envelope_method`` defaults to the production upper-envelope method
    and only exists so callers can override it (e.g. an upper-envelope benchmark)
    without duplicating this function. ``income_shock_batch_size`` is the same
    kind of override: left at None the solve interpolates all ``n_quad_points``
    income-shock draws at once, an integer dividing ``n_quad_points`` makes it
    work through them in blocks of that size, which lowers the peak memory of
    the interpolation step without changing the solution. The assets and experience grids are not
    configurable here: ``assets_end_of_period`` is always the production savings
    grid, and ``experience`` is always supplied per state-choice via
    ``continuous_grid_functions`` (``experience_grid_from_state``, wired in
    ``specify_model``).

    """
    sex_grid, edu_grid = specify_type_grids(
        sex_type=sex_type,
        edu_type=edu_type,
    )

    if sex_type == "all":
        # This is the best for H100.
        batch_seps = [44]  # Full model
        batch_mode = "period_max"
    else:
        batch_seps = [29, 43, 44]  # Fastest model single
        batch_mode = ["largest_block", "largest_block", "period_max", "largest_block"]

    continuous_states = {
        "assets_end_of_period": create_end_of_period_assets() / specs["wealth_unit"],
        # Supplied per state-choice by continuous_grid_functions
        # (experience_grid_from_state), wired in specify_model.
        "experience": None,
    }

    model_config = {
        "income_shock_batch_size": income_shock_batch_size,
        "min_period_batch_segments": batch_seps,
        "batch_mode": batch_mode,
        "n_periods": specs["n_periods"],
        "choices": np.arange(specs["n_choices"], dtype=int),
        "deterministic_states": {
            "education": edu_grid,
            "sex": sex_grid,
            "alg_1_claim": np.arange(3, dtype=int),
        },
        "stochastic_states": {
            "policy_state": np.arange(specs["n_policy_states"], dtype=int),
            "job_offer": np.arange(2, dtype=int),
            "partner_state": np.arange(specs["n_partner_states"], dtype=int),
            "health": np.arange(specs["n_all_health_states"], dtype=int),
        },
        "continuous_states": continuous_states,
        "n_quad_points": specs["n_quad_points"],
    }
    if upper_envelope_method is not None:
        model_config["upper_envelope"] = {"method": upper_envelope_method}
    return model_config


def cpu_build():
    """Place the model build phase on the CPU.

    The build -- state space, sparse stochastic transition map, batch information,
    and the data preparation that evaluates model functions over the observed
    states -- reserves no GPU memory this way. The arrays it creates stay
    uncommitted, so the solve and the likelihood still run on the default device
    and the arrays move there when they are first called.
    """
    return jax.default_device(jax.devices("cpu")[0])


def specify_model(
    path_dict,
    specs,
    subj_unc,
    custom_resolution_age,
    sim_specs=None,
    simulate_expectations=False,
    load_model=False,
    debug_info=None,
    sex_type="all",
    edu_type="all",
    util_type="add",
    upper_envelope_method=None,
    income_shock_batch_size=None,
):
    """Generate model class.

    ``upper_envelope_method`` and ``income_shock_batch_size`` let a caller override
    the production upper-envelope method and the income-shock block size; leave both
    at ``None`` for normal use.

    """

    SRA_belief_solution, specs = select_solution_transition_func_and_update_specs(
        specs=specs,
        subj_unc=subj_unc,
        custom_resolution_age=custom_resolution_age,
    )

    stochastic_states_transitions = {
        "policy_state": SRA_belief_solution,
        "job_offer": job_offer_process_transition,
        "partner_state": partner_transition,
        "health": health_transition,
    }

    model_config = create_model_config_wo_informed(
        specs=specs,
        sex_type=sex_type,
        edu_type=edu_type,
        upper_envelope_method=upper_envelope_method,
        income_shock_batch_size=income_shock_batch_size,
    )

    if sim_specs is not None:
        alternative_sim_specifications, specs = define_alternative_sim_specifications(
            path_dict=path_dict,
            sim_specs=sim_specs,
            simulate_expectations=simulate_expectations,
            specs=specs,
            custom_resolution_age=custom_resolution_age,
            subj_unc=subj_unc,
            sex_type=sex_type,
            edu_type=edu_type,
        )

    else:
        alternative_sim_specifications = None

    # Assign informed
    # Now we use the alternative sim specification to define informed in the solution
    # as deterministic state (type) and in the simulation as stochastic state.
    informed_states = np.arange(2, dtype=int)

    # First add it as a deterministic state
    model_config["deterministic_states"]["informed"] = informed_states

    if util_type == "add":
        from model_code.utility.utility_functions_add import create_utility_functions
    elif util_type == "cobb":
        from model_code.utility.utility_functions_cobb import create_utility_functions
    else:
        raise ValueError("unknown utility type")

    if specs["ERA_moves"]:
        era_append = "ERA_moves"
    else:
        era_append = "ERA_stays"

    if subj_unc:
        exp_append = "unc"
    else:
        exp_append = "no_unc"

    model_path = (
        path_dict["intermediate_data"]
        + f"model_{sex_type}_{edu_type}_{era_append}_{exp_append}.pkl"
    )

    # Sex- and age-specific experience grid, per state-choice (retired states
    # reuse the same axis as pension points; see experience_grid_from_state).
    continuous_grid_functions = {"experience": experience_grid_from_state}

    with cpu_build():
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
                continuous_grid_functions=continuous_grid_functions,
                model_load_path=model_path,
                alternative_sim_specifications=alternative_sim_specifications,
                debug_info=debug_info,
                use_stochastic_sparsity=True,
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
                continuous_grid_functions=continuous_grid_functions,
                model_save_path=model_path,
                alternative_sim_specifications=alternative_sim_specifications,
                debug_info=debug_info,
                use_stochastic_sparsity=True,
            )

    print("Model specified.", flush=True)
    return model


def define_alternative_sim_specifications(
    path_dict,
    sim_specs,
    simulate_expectations,
    specs,
    subj_unc,
    custom_resolution_age,
    sex_type,
    edu_type,
):
    stochastic_states_transitions = {
        "job_offer": job_offer_process_transition,
        "partner_state": partner_transition,
        "health": health_transition,
    }

    model_config = create_model_config_wo_informed(
        specs=specs,
        sex_type=sex_type,
        edu_type=edu_type,
    )

    # Now as stochastic in the sim objects
    model_config["stochastic_states"]["informed"] = np.arange(2, dtype=int)

    if simulate_expectations:

        def degenerate_informed_transition(informed):
            informed_prob = jax.lax.select(
                informed == 1,
                on_true=jnp.array([0.0, 1.0]),
                on_false=jnp.array([1.0, 0.0]),
            )
            return informed_prob

        stochastic_states_transitions["informed"] = degenerate_informed_transition

        transition_func_sim, specs = select_solution_transition_func_and_update_specs(
            specs=specs,
            subj_unc=subj_unc,
            custom_resolution_age=custom_resolution_age,
        )

    else:
        stochastic_states_transitions["informed"] = informed_transition

        transition_func_sim, specs = select_sim_policy_function_and_update_specs(
            specs=specs,
            subj_unc=subj_unc,
            announcement_age=sim_specs["announcement_age"],
            SRA_at_start=sim_specs["SRA_at_start"],
            SRA_at_retirement=sim_specs["SRA_at_retirement"],
            custom_resolution_age=custom_resolution_age,
        )

    stochastic_states_transitions["policy_state"] = transition_func_sim

    # Now specify the dict:
    alternative_sim_specifications = {
        "model_config": model_config,
        "stochastic_states_transitions": stochastic_states_transitions,
        "state_space_functions": create_state_space_functions(),
        "budget_constraint": budget_constraint,
        "shock_functions": shock_function_dict(),
    }
    return alternative_sim_specifications, specs


def specify_and_solve_model(
    path_dict,
    file_append,
    params,
    subj_unc,
    custom_resolution_age,
    load_model,
    load_solution,
    sim_specs=None,
    simulate_expectations=False,
    sex_type="all",
    edu_type="all",
    util_type="add",
    debug_info=None,
    chunked_solve=False,
    chunked_parallel=False,
):
    """Specify and solve model.

    Also includes possibility to save solutions.
    men_only=False,

    When ``chunked_solve`` is True (only valid for sex_type='all', edu_type='all'),
    the full solve is assembled from the four per-type (sex x edu) sub-model solves via
    ``dcegm.get_solve_from_small_models`` instead of one pooled solve, so the full
    solution never has to be solved on device in one piece. Loading a stored solution
    needs no solve and is unaffected. ``chunked_parallel`` dispatches the sub-model
    solves across ``jax.devices()`` (one block per device) rather than one at a time;
    it only helps with several devices and holds several blocks in memory at once.

    """

    specs = generate_derived_and_data_derived_specs(path_dict)

    # Generate model_specs
    model = specify_model(
        path_dict=path_dict,
        specs=specs,
        subj_unc=subj_unc,
        custom_resolution_age=custom_resolution_age,
        load_model=load_model,
        sim_specs=sim_specs,
        simulate_expectations=simulate_expectations,
        debug_info=debug_info,
        sex_type=sex_type,
        edu_type=edu_type,
        util_type=util_type,
    )

    # check if folder of model objects exits:
    solve_folder = get_model_results_path(path_dict, file_append)

    # Generate name of solution
    if subj_unc:
        resolution_age = specs["resolution_age"]
        sol_name = f"sol_subj_unc_{resolution_age}.pkl"
    else:
        sol_name = "sol_no_subj_unc.pkl"

    solution_file = solve_folder["solution"] + sol_name

    if chunked_solve and load_solution is not True:
        if not (sex_type == "all" and edu_type == "all"):
            raise ValueError(
                "chunked_solve=True is only valid for sex_type='all' and "
                "edu_type='all'; it assembles the full solve from per-type sub-models."
            )
        sub_models = [
            specify_model(
                path_dict=path_dict,
                specs=generate_derived_and_data_derived_specs(path_dict),
                subj_unc=subj_unc,
                custom_resolution_age=custom_resolution_age,
                load_model=load_model,
                sim_specs=sim_specs,
                simulate_expectations=simulate_expectations,
                debug_info=debug_info,
                sex_type=sub_sex_type,
                edu_type=sub_edu_type,
                util_type=util_type,
            )
            for sub_sex_type in ("men", "women")
            for sub_edu_type in ("low", "high")
        ]
        solve_from_small_models = dcegm.get_solve_from_small_models(
            small_models=sub_models,
            parallel=chunked_parallel,
            big_model=model,
        )
        model_solved = solve_from_small_models(params)
        if load_solution is False:
            pickle.dump(
                {
                    "value": model_solved.value,
                    "policy": model_solved.policy,
                    "endog_grid": model_solved.endog_grid,
                },
                open(solution_file, "wb"),
            )
        return model_solved

    if load_solution is None:
        model_solved = model.solve(params)
        return model_solved
    elif load_solution:
        model_solved = model.solve(params, load_sol_path=solution_file)
        return model_solved
    else:
        model_solved = model.solve(params, save_sol_path=solution_file)
        return model_solved


def specify_type_grids(sex_type, edu_type):
    if sex_type == "men":
        sex_grid = [0]
    elif sex_type == "women":
        sex_grid = [1]
    elif sex_type == "all":
        sex_grid = [0, 1]
    else:
        raise ValueError("sex_type not recognized")

    if edu_type == "all":
        edu_grid = [0, 1]
    elif edu_type == "low":
        edu_grid = [0]
    elif edu_type == "high":
        edu_grid = [1]
    else:
        raise ValueError("edu_type not recognized")

    return sex_grid, edu_grid
