import pandas as pd
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods

from model_code.specify_model import specify_and_solve_model, specify_model
from model_code.state_space.experience import construct_experience_years
from set_paths import get_model_resutls_path
from simulation.sim_tools.initial_conditions_sim import generate_start_states
from specs.derive_specs import read_and_derive_specs


def solve_and_simulate_scenario(
    path_dict,
    params,
    subj_unc,
    custom_resolution_age,
    SRA_at_start,
    SRA_at_retirement,
    annoucement_age,
    model_name,
    df_exists=True,
    only_informed=False,
    solution_exists=True,
    sol_model_exists=True,
    sim_model_exists=True,
):
    model_out_folder = get_model_resutls_path(path_dict, model_name)

    # Make intitial SRA only two digits after point

    df_name = create_df_name(
        path_dict=path_dict,
        custom_resolution_age=custom_resolution_age,
        only_informed=only_informed,
        annoucement_age=annoucement_age,
        SRA_at_start=SRA_at_start,
        SRA_at_retirement=SRA_at_retirement,
        subj_unc=subj_unc,
    )
    df_file = model_out_folder["simulation"] + df_name

    if df_exists:
        data_sim = pd.read_pickle(df_file)
        return data_sim

    # First we create the solution. As this is the expectation, the only
    # thing we need to know, if there is subjective uncertainty and if so
    # what the resolution age is (internal check for coherence)
    sol_container, model, params = specify_and_solve_model(
        path_dict=path_dict,
        params=params,
        file_append=model_name,
        custom_resolution_age=custom_resolution_age,
        subj_unc=subj_unc,
        load_model=sol_model_exists,
        load_solution=solution_exists,
    )

    model_params = model["options"]["model_params"]

    # In the simulation, things can be more difficult. First, suppose
    # agents hold subjective uncertainty. Then in the simulation, there can be two cases:
    # Smooth change of SRA(including no change) and announcment. Determine the relevant parameters
    # for this
    if subj_unc:
        # If there is no announcment, we have a smooth change with sim_alpha
        if annoucement_age is None:
            sim_alpha = (SRA_at_retirement - SRA_at_start) / (
                model_params["resolution_age"] - model_params["start_age"]
            )
            announcment_SRA = None
        else:
            sim_alpha = None
            announcment_SRA = SRA_at_retirement
    else:
        # If there is no uncertainty then we SRA at resolution is the same as SRA at start.
        # We also check that here
        if SRA_at_start != SRA_at_retirement:
            raise ValueError(
                "SRA at start and resolution must be the same when there is no uncertainty"
            )
        # Announcment is not allowed to be given
        if annoucement_age is not None:
            raise ValueError(
                "Announcment age can only be given in case of subjective uncertainty"
            )
        # We set sim_alpha to 0 for the simulation. (For the solution it is clear if subj_exp is False)
        sim_alpha = 0.0
        # We also set the announcment SRA to None
        announcment_SRA = None

    if df_exists:
        data_sim = pd.read_pickle(df_file)
        return data_sim
    else:
        data_sim = simulate_scenario(
            path_dict=path_dict,
            params=params,
            n_agents=model_params["n_agents"],
            seed=model_params["seed"],
            custom_resolution_age=custom_resolution_age,
            sim_alpha=sim_alpha,
            annoucement_age=annoucement_age,
            annoucement_SRA=announcment_SRA,
            initial_SRA=SRA_at_start,
            solution=sol_container,
            model_of_solution=model,
            sim_model_exists=sim_model_exists,
            only_informed=only_informed,
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
    custom_resolution_age,
    sim_alpha,
    annoucement_age,
    annoucement_SRA,
    initial_SRA,
    solution,
    model_of_solution,
    sim_model_exists,
    only_informed=False,
):
    model_sim, params = specify_model(
        path_dict=path_dict,
        params=params,
        subj_unc=False,
        custom_resolution_age=custom_resolution_age,
        sim_alpha=sim_alpha,
        annoucement_age=annoucement_age,
        annoucement_SRA=annoucement_SRA,
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
        only_informed=only_informed,
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
    df["income_wo_interest"] = df.groupby("agent")["wealth_at_beginning"].shift(
        -1
    ) - df["savings"] * (1 + params["interest_rate"])

    # periodic savings and savings rate
    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["savings_rate"] = df["savings_dec"] / df["total_income"]

    # Create lagged health state to filter out already dead people
    df["health_lag"] = df.groupby("agent")["health"].shift(1)
    df = df[(df["health"] != 2) | ((df["health"] == 2) & (df["health_lag"] != 2))]

    return df


def create_df_name(
    path_dict,
    custom_resolution_age,
    annoucement_age,
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
        if annoucement_age is None:
            df_name = f"df_subj_unc_{resolution_age}_{SRA_at_start}_{SRA_at_retirement}_{name_append}"
        else:
            df_name = f"df_subj_unc_{resolution_age}_{SRA_at_start}_{SRA_at_retirement}_{annoucement_age}_{name_append}"
    else:
        df_name = f"df_no_unc_{SRA_at_retirement}_{name_append}"
    return df_name
