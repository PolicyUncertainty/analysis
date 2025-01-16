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
    sol_model_exists=True,
    sim_model_exists=True,
):
    solution_est, model, params = specify_and_solve_model(
        path_dict=path_dict,
        params=params,
        update_spec_for_policy_state=solve_update_specs_func,
        policy_state_trans_func=solve_policy_trans_func,
        file_append=file_append_sol,
        load_model=sol_model_exists,
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
        solution=solution_est,
        model_of_solution=model,
        sim_model_exists=sim_model_exists,
    )
    data_sim["exp_years"] = data_sim["experience"] * (
        model_params["max_init_experience"] + data_sim.index.get_level_values("period")
    )
    return data_sim


def simulate_scenario(
    path_dict,
    n_agents,
    seed,
    params,
    update_spec_for_policy_state,
    policy_state_func_scenario,
    solution,
    model_of_solution,
    sim_model_exists,
):
    model_sim, params = specify_model(
        path_dict=path_dict,
        params=params,
        update_spec_for_policy_state=update_spec_for_policy_state,
        policy_state_trans_func=policy_state_func_scenario,
        load_model=sim_model_exists,
        model_type="simulation",
    )

    options = model_of_solution["options"]

    initial_states, wealth_agents = generate_start_states(
        path_dict, params, model_of_solution, n_agents, seed
    )

    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        wealth_initial=wealth_agents,
        n_periods=options["model_params"]["n_periods"],
        params=params,
        seed=seed,
        endog_grid_solved=solution["endog_grid"],
        value_solved=solution["value"],
        policy_solved=solution["policy"],
        model=model_of_solution,
        model_sim=model_sim,
    )
    df = create_simulation_df(sim_dict)
    df["age"] = (
        df.index.get_level_values("period") + options["model_params"]["start_age"]
    )
    # Create lagged health state for each agent and period
    df["health_lag"] = df.groupby("agent")["health"].shift(1)
    # Filter out individuals for which health state and health lag is 2
    df = df[(df["health"] != 2) | ((df["health"] == 2) & (df["health_lag"] != 2))]

    df["wealth_at_beginning"] = df["savings"] + df["consumption"]
    # Create income var by shifting period of 1 of individuals and then substract
    # savings from resoures at beginning of period
    df["labor_income"] = df.groupby("agent")["wealth_at_beginning"].shift(-1) - df[
        "savings"
    ] * (1 + params["interest_rate"])
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["age"] = (
        df.index.get_level_values("period") + options["model_params"]["start_age"]
    )
    return df
