import pandas as pd
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from model_code.specify_model import specify_and_solve_model
from model_code.specify_model import specify_model
from simulation.sim_tools.initial_conditions_sim import generate_start_states


def solve_and_simulate_scenario(
    path_dict,
    params,
    solve_update_specs_func,
    solve_policy_trans_func,
    simulate_update_specs_func,
    simulate_policy_trans_func,
    solution_exists,
    file_append_sol,
    model_exists,
):
    model_solution_est, model, params = specify_and_solve_model(
        path_dict=path_dict,
        params=params,
        update_spec_for_policy_state=solve_update_specs_func,
        policy_state_trans_func=solve_policy_trans_func,
        file_append=file_append_sol,
        load_model=model_exists,
        load_solution=solution_exists,
    )
    model_params = model["options"]["model_params"]

    data_sim = simulate_scenario(
        path_dict=path_dict,
        params=params,
        n_agents=model_params["n_agents"],
        seed=model_params["seed"],
        update_spec_for_policy_state=simulate_update_specs_func,
        policy_state_func_scenario=simulate_policy_trans_func,
        expected_model=model_solution_est,
    )
    return data_sim


def simulate_scenario(
    path_dict,
    n_agents,
    seed,
    params,
    update_spec_for_policy_state,
    policy_state_func_scenario,
    expected_model,
):
    # Generate dcegm model for project specs
    model, params = specify_model(
        path_dict=path_dict,
        params=params,
        update_spec_for_policy_state=update_spec_for_policy_state,
        policy_state_trans_func=policy_state_func_scenario,
        load_model=True,
    )

    options = model["options"]

    initial_states, wealth_agents = generate_start_states(
        path_dict, params, model, n_agents, seed
    )

    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=wealth_agents,
        n_periods=options["model_params"]["n_periods"],
        params=params,
        seed=seed,
        endog_grid_solved=expected_model["endog_grid"],
        value_solved=expected_model["value"],
        policy_solved=expected_model["policy"],
        model=model,
    )
    df = create_simulation_df(sim_dict)
    df["age"] = (
        df.index.get_level_values("period") + options["model_params"]["start_age"]
    )

    df["wealth_at_beginning"] = df["savings"] + df["consumption"]
    # Create income var by shifting period of 1 of individuals and then substract
    # savings from resoures at beginning of period
    df["labor_income"] = df.groupby("agent")["wealth_at_beginning"].shift(-1) - df[
        "savings"
    ] * (1 + params["interest_rate"])
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["age"] = (
        df.index.get_level_values("period") + options["model_params"]["start_age"]
    )

    return df
