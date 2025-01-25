import pandas as pd
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from model_code.policy_processes.select_policy_belief import (
    select_expectation_functions_and_model_sol_names,
)
from model_code.specify_model import specify_and_solve_model
from model_code.specify_model import specify_model
from model_code.state_space import construct_experience_years
from set_paths import get_model_resutls_path
from simulation.sim_tools.initial_conditions_sim import generate_start_states


def solve_and_simulate_scenario(
    path_dict,
    params,
    expected_alpha,
    sim_alpha,
    initial_SRA,
    resolution,
    model_name,
    df_exists,
    solution_exists,
    sol_model_exists=True,
    sim_model_exists=True,
):
    solution_est, model, params = specify_and_solve_model(
        path_dict=path_dict,
        params=params,
        file_append=model_name,
        resolution=resolution,
        expected_alpha=expected_alpha,
        load_model=sol_model_exists,
        load_solution=solution_exists,
    )

    model_params = model["options"]["model_params"]
    (
        update_funcs,
        transition_funcs,
        model_sol_names,
    ) = select_expectation_functions_and_model_sol_names(
        path_dict,
        expected_alpha=expected_alpha,
        sim_alpha=sim_alpha,
        resolution=resolution,
    )

    solve_folder = get_model_resutls_path(path_dict, model_name)

    # Make intitial SRA only two digits after point
    df_file = (
        f"sra_"
        + "{:.2f}".format(expected_alpha)
        + solve_folder["simulation"]
        + model_sol_names["simulation"]
    )
    if df_exists:
        data_sim = pd.read_pickle(df_file)
        return data_sim
    else:
        data_sim = simulate_scenario(
            path_dict=path_dict,
            params=params,
            n_agents=model_params["n_agents"],
            seed=model_params["seed"],
            update_spec_for_policy_state=update_funcs["simulation"],
            policy_state_func_scenario=transition_funcs["simulation"],
            initial_SRA=initial_SRA,
            solution=solution_est,
            model_of_solution=model,
            sim_model_exists=sim_model_exists,
        )
        if df_exists is None:
            return data_sim
        else:
            data_sim.to_pickle(df_file)
            return data_sim


def simulate_scenario(
    path_dict,
    n_agents,
    seed,
    params,
    update_spec_for_policy_state,
    policy_state_func_scenario,
    initial_SRA,
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
        path_dict=path_dict,
        params=params,
        model=model_of_solution,
        inital_SRA=initial_SRA,
        n_agents=n_agents,
        seed=seed,
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

    # Create additional variables
    model_params = options["model_params"]
    df["age"] = df.index.get_level_values("period") + model_params["start_age"]
    # Create experience years
    df["exp_years"] = construct_experience_years(
        experience=df["experience"].values,
        period=df.index.get_level_values("period").values,
        max_exp_diffs_per_period=model_params["max_exp_diffs_per_period"],
    )

    # Create policy value
    df["policy_state_value"] = (
        model_params["min_SRA"] + df["policy_state"] * model_params["SRA_grid_size"]
    )

    # Assign working hours for choice 1
    df["working_hours"] = 0.0
    for sex_var in range(model_params["n_sexes"]):
        for edu_var in range(model_params["n_education_types"]):
            df.loc[
                (df["choice"] == 3)
                & (df["sex"] == sex_var)
                & (df["education"] == edu_var),
                "working_hours",
            ] = model_params["av_annual_hours_ft"][sex_var, edu_var]

            df.loc[
                (df["choice"] == 2)
                & (df["sex"] == sex_var)
                & (df["education"] == edu_var),
                "working_hours",
            ] = model_params["av_annual_hours_pt"][sex_var, edu_var]

    # Create income vars:
    # First wealth at the beginning of period as the sum of savings and consumption
    df["wealth_at_beginning"] = df["savings"] + df["consumption"]
    # Then total income as the difference between wealth at the beginning of next period and savings
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    # Finally the savings decision
    df["savings_dec"] = df["total_income"] - df["consumption"]

    # Create lagged health state to filter out already dead people
    df["health_lag"] = df.groupby("agent")["health"].shift(1)
    df = df[(df["health"] != 2) | ((df["health"] == 2) & (df["health_lag"] != 2))]

    return df
