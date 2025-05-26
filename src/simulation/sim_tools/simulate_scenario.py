import pandas as pd

from model_code.specify_model import specify_and_solve_model
from model_code.state_space.experience import construct_experience_years
from set_paths import get_model_resutls_path
from simulation.sim_tools.start_obs_for_sim import generate_start_states_from_obs
from specs.derive_specs import read_and_derive_specs


def solve_and_simulate_scenario(
    path_dict,
    params,
    subj_unc,
    custom_resolution_age,
    SRA_at_start,
    SRA_at_retirement,
    announcement_age,
    model_name,
    df_exists=True,
    only_informed=False,
    solution_exists=True,
    sol_model_exists=True,
):
    model_out_folder = get_model_resutls_path(path_dict, model_name)

    # Make intitial SRA only two digits after point

    df_name = create_df_name(
        path_dict=path_dict,
        custom_resolution_age=custom_resolution_age,
        only_informed=only_informed,
        announcement_age=announcement_age,
        SRA_at_start=SRA_at_start,
        SRA_at_retirement=SRA_at_retirement,
        subj_unc=subj_unc,
    )
    df_file = model_out_folder["simulation"] + df_name

    if df_exists:
        data_sim = pd.read_pickle(df_file)
        return data_sim

    # Create model and assign simulation specs.
    sim_specs = {
        "announcement_age": announcement_age,
        "SRA_at_start": SRA_at_start,
        "SRA_at_retirement": SRA_at_retirement,
    }
    model_solved = specify_and_solve_model(
        path_dict=path_dict,
        params=params,
        file_append=model_name,
        custom_resolution_age=custom_resolution_age,
        subj_unc=subj_unc,
        load_model=sol_model_exists,
        load_solution=solution_exists,
        sim_specs=sim_specs,
    )

    if df_exists:
        data_sim = pd.read_pickle(df_file)
        return data_sim
    else:
        data_sim = simulate_scenario(
            path_dict=path_dict,
            initial_SRA=SRA_at_start,
            model_solved=model_solved,
            only_informed=only_informed,
        )
        if df_exists is None:
            return data_sim
        else:
            data_sim.to_pickle(df_file)
            return data_sim


def simulate_scenario(
    path_dict,
    model_solved,
    initial_SRA,
    only_informed=False,
):

    # initial_states, wealth_agents = draw_initial_states(
    #     path_dict=path_dict,
    #     params=params,
    #     model=model_of_solution,
    #     inital_SRA=initial_SRA,
    #     seed=seed,
    #     only_informed=only_informed,
    # )

    initial_states, initial_wealth = generate_start_states_from_obs(
        path_dict=path_dict,
        params=model_solved.params,
        model_solved=model_solved,
        inital_SRA=initial_SRA,
        only_informed=only_informed,
    )
    model_specs = model_solved.model_specs

    df = model_solved.simulate(
        initial_states=initial_states,
        seed=model_specs["seed"],
    )

    # Create additional variables
    df["age"] = df.index.get_level_values("period") + model_specs["start_age"]
    # Create experience years
    df["exp_years"] = construct_experience_years(
        experience=df["experience"].values,
        period=df.index.get_level_values("period").values,
        max_exp_diffs_per_period=model_specs["max_exp_diffs_per_period"],
    )

    # Create policy value
    df["policy_state_value"] = (
        model_specs["min_SRA"] + df["policy_state"] * model_specs["SRA_grid_size"]
    )

    # Assign working hours for choice 1
    df["working_hours"] = 0.0
    for sex_var in range(model_specs["n_sexes"]):
        for edu_var in range(model_specs["n_education_types"]):
            df.loc[
                (df["choice"] == 3)
                & (df["sex"] == sex_var)
                & (df["education"] == edu_var),
                "working_hours",
            ] = model_specs["av_annual_hours_ft"][sex_var, edu_var]

            df.loc[
                (df["choice"] == 2)
                & (df["sex"] == sex_var)
                & (df["education"] == edu_var),
                "working_hours",
            ] = model_specs["av_annual_hours_pt"][sex_var, edu_var]

    # Create income vars:
    # First wealth at the beginning of period as the sum of savings and consumption
    df["wealth_at_beginning"] = df["savings"] + df["consumption"]
    # Then total income as the difference between wealth at the beginning of next period and savings
    df["total_income"] = (
        df.groupby("agent")["wealth_at_beginning"].shift(-1) - df["savings"]
    )
    df["income_wo_interest"] = df.groupby("agent")["wealth_at_beginning"].shift(
        -1
    ) - df["savings"] * (1 + model_specs["interest_rate"])

    # periodic savings and savings rate
    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["savings_rate"] = df["savings_dec"] / df["total_income"]

    # Create lagged health state to filter out already dead people
    df["health_lag"] = df.groupby("agent")["health"].shift(1)
    df = df[(df["health"] != 3) | ((df["health"] == 3) & (df["health_lag"] != 3))]

    return df


def create_df_name(
    path_dict,
    custom_resolution_age,
    announcement_age,
    only_informed,
    SRA_at_start,
    SRA_at_retirement,
    subj_unc,
):
    # Create df name
    if only_informed:
        name_append = "debiased.pkl"
    else:
        name_append = "biased.pkl"

    if custom_resolution_age is None:
        specs = read_and_derive_specs(path_dict["specs"])
        resolution_age = specs["resolution_age_estimation"]
    else:
        resolution_age = custom_resolution_age

    if subj_unc:
        if announcement_age is None:
            df_name = f"df_subj_unc_{resolution_age}_{SRA_at_start}_{SRA_at_retirement}_{name_append}"
        else:
            df_name = f"df_subj_unc_{resolution_age}_{SRA_at_start}_{SRA_at_retirement}_{announcement_age}_{name_append}"
    else:
        df_name = f"df_no_unc_{SRA_at_retirement}_{name_append}"
    return df_name
