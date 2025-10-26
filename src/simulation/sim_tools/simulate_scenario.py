import pandas as pd

from model_code.pension_system.early_retirement_paths import check_very_long_insured
from model_code.specify_model import (
    define_alternative_sim_specifications,
    specify_and_solve_model,
)
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
    initial_states=None,
    df_exists=True,
    only_informed=False,
    solution_exists=True,
    sol_model_exists=True,
    model_solution=None,
    sex_type="all",
    edu_type="all",
    util_type="add",
):
    """
    Solve and simulate a policy scenario for the retirement model.

    This function either loads existing results or creates new ones by solving the model
    and simulating agent behavior under specified policy conditions.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing all project paths
    params : dict
        Estimated model parameters
    subj_unc : bool
        Whether agents face subjective uncertainty about future SRA
    custom_resolution_age : int, optional
        Age at which uncertainty resolves (None uses spec default)
    SRA_at_start : int
        Initial statutory retirement age at model start
    SRA_at_retirement : float
        Final statutory retirement age (what SRA becomes)
    announcement_age : int, optional
        Age at which policy change is announced (None = no announcement)
    model_name : str
        Model identifier for file naming

    Loading Flags
    -------------
    df_exists : bool or None
        - True: Load existing simulation DataFrame, error if not found
        - False: Create new simulation DataFrame and save it
        - None: Create new simulation DataFrame but don't save
    only_informed : bool, default False
        Whether to simulate only informed agents (True) or include misinformed (False)
    solution_exists : bool, default True
        Whether to load existing model solution (True) or solve from scratch (False)
    sol_model_exists : bool, default True
        Whether to load existing model specification (True) or create new (False)
    model_solution : object, optional
        Pre-solved model object to reuse (None = solve new or load from disk)

    Demographics
    ------------
    sex_type : str, default "all"
        Which gender to simulate ("all", "male", "female")
    edu_type : str, default "all"
        Which education level to simulate ("all", "low", "high")
    util_type : str, default "add" = additive separable
        Utility function specification

    Returns
    -------
    df : pd.DataFrame
        Simulated lifecycle data for all agents
    model_solved : object
        Solved model object (for reuse in subsequent calls)

    Notes
    -----
    Loading flags control computational efficiency:
    - Set solution_exists=True and pass model_solution from previous call to reuse solutions
    - Set df_exists=True to skip simulation if results already computed. !This will return None for model_solution!
    - Use df_exists=None for temporary simulations you don't want to save
    """
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

    if df_exists:
        data_sim = pd.read_csv(df_file)
        print(
            "Loading existing simulated dataframe from file. Warning: returns None for model solution."
        )
        return data_sim, None

    if model_solution is None:
        # Create and solve model
        model_solved = specify_and_solve_model(
            path_dict=path_dict,
            params=params,
            file_append=model_name,
            custom_resolution_age=custom_resolution_age,
            subj_unc=subj_unc,
            load_model=sol_model_exists,
            load_solution=solution_exists,
            sim_specs=sim_specs,
            sex_type=sex_type,
            edu_type=edu_type,
            util_type=util_type,
        )
    else:
        # Use existing model solution but update sim specs
        specs = generate_derived_and_data_derived_specs(path_dict)

        model_solved = model_solution
        alternative_sim_specifications, alternative_sim_specs = (
            define_alternative_sim_specifications(
                sim_specs=sim_specs,
                specs=specs,
                subj_unc=subj_unc,
                custom_resolution_age=model_solved.model_specs["resolution_age"],
                sex_type=sex_type,
                edu_type=edu_type,
            )
        )
        model_solved.set_alternative_sim_funcs(
            alternative_sim_specifications=alternative_sim_specifications,
            alternative_specs=alternative_sim_specs,
        )
    # Simulate
    data_sim = simulate_scenario(
        path_dict=path_dict,
        initial_SRA=SRA_at_start,
        model_solved=model_solved,
        only_informed=only_informed,
        initial_states=initial_states,
    )
    if df_exists is None:
        # do not save df
        return data_sim, model_solved
    else:
        # save df
        data_sim.to_csv(df_file)
        return data_sim, model_solved


def simulate_scenario(
    path_dict,
    model_solved,
    initial_SRA,
    only_informed=False,
    initial_states=None,
):
    if initial_states is None:
        # Generate initial states from observed data
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
    # Kick out dead people
    df = df[df["health"] != 3].copy()
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
    df = df.copy()

    # Create income vars:
    # First, total income as the difference between wealth at the beginning of next period and savings
    df.loc[:, "total_income"] = df["assets_begin_of_period"] - df.groupby("agent")[
        "savings"
    ].shift(1)

    # periodic savings and savings rate
    df.loc[:, "savings_dec"] = df["total_income"] - df["consumption"]
    df.loc[:, "savings_rate"] = df["savings_dec"] / df["total_income"]

    # Create gross own income (without pension income)
    df.loc[:, "gross_own_income"] = (
        (df["choice"] == 0) * df["gross_retirement_income"]  # Retired
        + (df["choice"] == 1) * 0  # Unemployed
        + ((df["choice"] == 2) | (df["choice"] == 3))
        * df["gross_labor_income"]  # Part-time or full-time work
    )
    return df


def _transform_states_into_variables(df, specs):
    """Transform state variables into more interpretable variables."""
    df = df.copy()

    # Create additional variables
    df.loc[:, "age"] = df["period"] + specs["start_age"]

    # Create experience years
    df.loc[:, "exp_years"] = construct_experience_years(
        float_experience=df["experience"].values,
        period=df["period"].values,
        is_retired=df["lagged_choice"].values == 0,
        model_specs=specs,
    )

    # Create policy value
    df.loc[:, "policy_state_value"] = (
        specs["min_SRA"] + df["policy_state"] * specs["SRA_grid_size"]
    )
    return df


def _compute_working_hours(df, specs):
    """Compute working hours based on employment choice and demographics."""
    df = df.copy()

    # Initialize working_hours column
    df.loc[:, "working_hours"] = 0.0

    for sex_var in [0, 1]:
        for edu_var in range(specs["n_education_types"]):
            # Full-time work
            mask_ft = (
                (df["choice"] == 3)
                & (df["sex"] == sex_var)
                & (df["education"] == edu_var)
            )
            df.loc[mask_ft, "working_hours"] = specs["av_annual_hours_ft"][
                sex_var, edu_var
            ]

            # Part-time work
            mask_pt = (
                (df["choice"] == 2)
                & (df["sex"] == sex_var)
                & (df["education"] == edu_var)
            )
            df.loc[mask_pt, "working_hours"] = specs["av_annual_hours_pt"][
                sex_var, edu_var
            ]

    return df


def _compute_actual_retirement_age(df):
    """Compute actual retirement age based on choice variable."""
    df = df.copy()

    df_retirement = df[df["choice"] == 0].copy()
    actual_retirement_ages = df_retirement.groupby("agent")["age"].min()
    df.loc[:, "actual_retirement_age"] = df["agent"].map(actual_retirement_ages)
    return df


def _compute_initial_informed_status(df, specs):
    """Compute initial informed status based on informed state variable."""
    df = df.copy()

    df_initial = df[df["period"] == 0][["agent", "informed"]].copy()
    informed_status = df_initial.set_index("agent")["informed"].map(
        {0: "Uninformed", 1: "Informed"}
    )
    df.loc[:, "initial_informed"] = df["agent"].map(informed_status)
    return df


def _add_very_long_insured_claim(df, specs):
    """Add a column indicating whether the individual is classified as 'very long insured'."""
    retirement_age_difference = df["policy_state_value"] - df["age"]

    df["very_long_insured"] = check_very_long_insured(
        retirement_age_difference=retirement_age_difference.values,
        experience_years=df["exp_years"].values,
        policy_state=df["policy_state"].values,
        sex=df["sex"].values.astype(int),
        model_specs=specs,
    )
    return df


def create_additional_variables(df, specs):
    """Wrapper function to create additional variables in the simulated dataframe."""
    df = df.copy()

    df = _create_income_variables(df, specs)
    df = _transform_states_into_variables(df, specs)
    df = _compute_working_hours(df, specs)
    df = _compute_actual_retirement_age(df)
    df = _add_very_long_insured_claim(df, specs)
    df = _compute_initial_informed_status(df, specs)
    return df
