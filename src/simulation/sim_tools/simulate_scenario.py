import pandas as pd

from model_code.specify_model import specify_and_solve_model
from model_code.state_space.experience import construct_experience_years
from set_paths import get_model_results_path
from simulation.sim_tools.start_obs_for_sim import generate_start_states_from_obs
from specs.derive_specs import (
    generate_derived_and_data_derived_specs,
    read_and_derive_specs,
)


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
    model_out_folder = get_model_results_path(path_dict, model_name)

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
        sex_type="all",
        edu_type="all",
    )

    if df_exists:
        data_sim = pd.read_csv(df_file)
        return data_sim, model_solved
    else:
        data_sim = simulate_scenario(
            path_dict=path_dict,
            initial_SRA=SRA_at_start,
            model_solved=model_solved,
            only_informed=only_informed,
        )
        if df_exists is None:
            return data_sim, model_solved
        else:
            data_sim.to_csv(df_file)
            return data_sim, model_solved


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

    initial_states = generate_start_states_from_obs(
        path_dict=path_dict,
        params=model_solved.params,
        model_class=model_solved,
        inital_SRA=initial_SRA,
        only_informed=only_informed,
    )
    specs = generate_derived_and_data_derived_specs(path_dict)

    df = model_solved.simulate(
        states_initial=initial_states,
        seed=specs["seed"],
    )
    df.reset_index(inplace=True)   
    df = create_additional_variables(df, specs)
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
        name_append = "debiased.csv"
    else:
        name_append = "biased.csv"

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




def _create_income_variables(df, specs):
    """Create income related variables in the simulated dataframe.
    Note: check budget equation first! They may already be there (under "aux").
    """
    # Create income vars:
    # First, total income as the difference between wealth at the beginning of next period and savings
    df["total_income"] = df["assets_begin_of_period"] - df.groupby("agent")[
        "savings"
    ].shift(1)

    # periodic savings and savings rate
    df["savings_dec"] = df["total_income"] - df["consumption"]
    df["savings_rate"] = df["savings_dec"] / df["total_income"]

    # Create lagged health state to filter out already dead people
    df["lagged_health"] = df.groupby("agent")["health"].shift(1)
    df = df[(df["health"] != 3) | ((df["health"] == 3) & (df["lagged_health"] != 3))]

    # Create gross own income (without pension income)
    df["gross_own_income"] = (
        (df["choice"] == 0) * df["gross_retirement_income"] +  # Retired
        (df["choice"] == 1) * 0 +  # Unemployed
        ((df["choice"] == 2) | (df["choice"] == 3)) * df["gross_labor_income"]  # Part-time or full-time work
    )
    return df

def _transform_states_into_variables(df, specs):
    """Transform state variables into more interpretable variables."""
    # Create additional variables
    df["age"] = df.index.get_level_values("period") + specs["start_age"]
    # Create experience years
    df["exp_years"] = construct_experience_years(
        float_experience=df["experience"].values,
        period=df.index.get_level_values("period").values,
        is_retired=df["lagged_choice"].values == 0,
        model_specs=specs,
    )

    # Create policy value
    df["policy_state_value"] = (
        specs["min_SRA"] + df["policy_state"] * specs["SRA_grid_size"]
    )
    return df

def _compute_working_hours(df, specs):
    """Compute working hours based on employment choice and demographics."""
    df["working_hours"] = 0.0
    for sex_var in [0, 1]:
        for edu_var in range(specs["n_education_types"]):
            df.loc[
                (df["choice"] == 3)
                & (df["sex"] == sex_var)
                & (df["education"] == edu_var),
                "working_hours",
            ] = specs["av_annual_hours_ft"][sex_var, edu_var]

            df.loc[
                (df["choice"] == 2)
                & (df["sex"] == sex_var)
                & (df["education"] == edu_var),
                "working_hours",
            ] = specs["av_annual_hours_pt"][sex_var, edu_var]
    return df

def _compute_actual_retirement_age(df):
    """Compute actual retirement age based on choice variable."""
    df_retirement = df[df["choice"] == 0]
    actual_retirement_ages = df_retirement.groupby("agent")["age"].min()
    df["actual_retirement_age"] = df["agent"].map(actual_retirement_ages)
    return df

def _compute_initial_informed_status(df, specs):
    """Compute initial informed status based on informed state variable."""
    df_initial = df[df.index.get_level_values("period") == 0]
    informed_status = df_initial.set_index("agent")["informed"]
    informed_status = informed_status.map({0: "Uninformed", 1: "Informed"})
    df["initial_informed"] = df["agent"].map(informed_status)
    return df

def create_additional_variables(df, specs):
    """Wrapper function to create additional variables in the simulated dataframe."""
    df = _create_income_variables(df, specs)
    df = _transform_states_into_variables(df, specs)
    df = _compute_working_hours(df, specs)
    df = _compute_actual_retirement_age(df)
    df = _compute_initial_informed_status(df, specs)
    return df