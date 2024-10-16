import os

import numpy as np
import pandas as pd
from process_data.sample_creation_scripts.data_tools import filter_below_age
from process_data.sample_creation_scripts.data_tools import filter_by_sex
from process_data.sample_creation_scripts.data_tools import filter_est_years
from process_data.sample_creation_scripts.partner_code import create_partner_state
from process_data.var_resources.soep_vars import create_choice_variable
from process_data.var_resources.soep_vars import create_education_type
from process_data.var_resources.soep_vars import create_experience_variable
from process_data.var_resources.soep_vars import generate_job_separation_var
from process_data.var_resources.wealth import add_wealth


def create_structural_est_sample(paths, load_data=False, options=None):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "structural_estimation_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Export parameters from options
    start_year = options["start_year"]
    end_year = options["end_year"]
    min_ret_age = options["min_ret_age"]
    start_age = options["start_age"]

    # Load and merge data state data from SOEP core (all but wealth)
    df = load_and_merge_soep_core(soep_c38_path=paths["soep_c38"])

    # Create partner state(Merges also partners)
    df = create_partner_state(df, filter_missing=True)

    # (labor) choice
    df = create_choice_variable(df)

    # filter data. Leave additional years in for lagging and leading. For now no women
    df = filter_data(df, start_year, end_year, start_age, no_women=True)

    # Job separation
    df = generate_job_separation_var(df)

    # lagged choice
    df = create_lagged_and_lead_variables(df, start_year, end_year, start_age)

    # Add wealth data
    df = add_wealth(df, paths, options)
    # df.to_pickle(paths["intermediate_data"] + "wealth_sample.pkl")

    # Now create more observed choice variables
    # period
    df["period"] = df["age"] - start_age

    # policy_state
    df["policy_state"] = create_policy_state(df["gebjahr"])

    (
        df["policy_state_value"],
        df["policy_state"],
    ) = modify_policy_state(df["policy_state"], options)

    # experience
    df = create_experience_variable(df)

    # education
    df = create_education_type(df)

    # additional restrictions based on model setup
    df = enforce_model_choice_restriction(df, min_ret_age, options["max_ret_age"])

    # Construct job offer state
    df = determine_observed_job_offers(df)

    # Keep relevant columns (i.e. state variables) and set their minimal datatype
    type_dict = {
        "period": "int8",
        "choice": "int8",
        "lagged_choice": "int8",
        "policy_state": "int8",
        "policy_state_value": "float32",
        "partner_state": "int8",
        "job_offer": "int8",
        "experience": "int8",
        "wealth": "float32",
        "education": "int8",
        "full_observed_state": "bool",
    }
    df = df[list(type_dict.keys())]
    df = df.astype(type_dict)

    print(
        str(len(df))
        + " observations in final structural estimation dataset. \n ----------------"
    )

    # Anonymize and save data
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(out_file_path)

    # save data
    df.to_pickle(out_file_path)

    return df


def determine_observed_job_offers(data):
    """Determine if a job offer is observed and if so what it is. The function
    implements the following rule:

    Assume lagged choice equal to 1 (working), then the state is fully observed:
        - If choice equal 1 (continued working), then there is a job offer, i.e. equal to 1
        - If choice is unemployed (0) or retired (2) and you got fired then job offer equal 0
        - Same as before, but not fired then job offer equal to 1

    Assume lagged choice equal to 0 (unemployed), then the state is partially observed:
        - If choice is working, then the state is fully observed and there is a job offer
        - If choice is different, then one is not observed

    Lagged choice equal to 2(retired), will be dropped as only choice equal to 2 is allowed

    Therefore the unobserved job offer states are, where individuals are unemployed and remain unemployed or retire.

    """
    data["job_offer"] = -99
    data["full_observed_state"] = False

    # Individuals working have job offer equal to 1 and are fully observed
    data.loc[data["choice"] == 1, "job_offer"] = 1
    data.loc[data["choice"] == 1, "full_observed_state"] = True

    # Individuals who are unemployed or retird and are fired this period have job offer
    # equal to 0. This includes individuals with lagged choice unemployment, as they
    # might be interviewed after firing.
    # Update: Use only employed people. Talk about that!!!
    maskfired = (
        (data["choice"].isin([0, 2]))
        & (data["job_sep_this_year"] == 1)
        & (data["lagged_choice"] == 1)
    )
    data.loc[maskfired, "job_offer"] = 0
    data.loc[maskfired, "full_observed_state"] = True

    # Everybody who was not fired is also fully observed an has an job offer
    mask_not_fired = (
        (data["choice"].isin([0, 2]))
        & (data["job_sep_this_year"] == 0)
        & (data["lagged_choice"] == 1)
    )
    data.loc[mask_not_fired, "job_offer"] = 1
    data.loc[mask_not_fired, "full_observed_state"] = True

    return data


def load_and_merge_soep_core(soep_c38_path):
    # Load SOEP core data
    pgen_data = pd.read_stata(
        f"{soep_c38_path}/pgen.dta",
        columns=[
            "syear",
            "pid",
            "hid",
            "pgemplst",
            "pgexpft",
            "pgexppt",
            "pgstib",
            "pgpartz",
            "pglabgro",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )
    ppathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["pid", "hid", "syear", "sex", "gebjahr", "parid", "rv_id"],
        convert_categoricals=False,
    )
    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )

    # Add pl data
    pl_data_reader = pd.read_stata(
        f"{soep_c38_path}/pl.dta",
        columns=["pid", "hid", "syear", "plb0304_h"],
        chunksize=100000,
        convert_categoricals=False,
    )
    pl_data = pd.DataFrame()
    for itm in pl_data_reader:
        pl_data = pd.concat([pl_data, itm])
    merged_data = pd.merge(
        merged_data, pl_data, on=["pid", "hid", "syear"], how="inner"
    )

    # Now get household level data
    hl_data = pd.read_stata(
        f"{soep_c38_path}/hl.dta",
        columns=["hid", "syear", "hlc0043"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, hl_data, on=["hid", "syear"], how="left")

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data


def filter_data(merged_data, start_year, end_year, start_age, no_women=True):
    """This function filters the data according to the model setup.

    Specifically, it filters out young people, women (if no_women=True), and years
    outside of estimation range. It leaves one year younger and one year below in the
    sample to construct lagged_choice.

    """
    merged_data = filter_below_age(merged_data, start_age - 1)

    merged_data = filter_by_sex(merged_data, no_women=no_women)

    merged_data = filter_est_years(merged_data, start_year, end_year)
    return merged_data


def create_lagged_and_lead_variables(merged_data, start_year, end_year, start_age):
    """This function creates the lagged choice variable and drops missing lagged
    choices."""
    # Create full index with all possible combinations of pid and syear. Otherwise if
    # we just shift the data, people having missing years in their observations get
    # assigned variables from multi years back.
    pid_indexes = merged_data.index.get_level_values(0).unique()
    full_index = pd.MultiIndex.from_product(
        [pid_indexes, range(start_year - 1, end_year + 2)],
        names=["pid", "syear"],
    )
    full_container = pd.DataFrame(
        index=full_index, data=np.nan, dtype=float, columns=merged_data.columns
    )
    full_container.update(merged_data)
    full_container["lagged_choice"] = full_container.groupby(["pid"])["choice"].shift()
    full_container["job_sep_this_year"] = full_container.groupby(["pid"])[
        "job_sep"
    ].shift(-1)
    merged_data = full_container[full_container["lagged_choice"].notna()]
    merged_data = merged_data[merged_data["job_sep_this_year"].notna()]

    # We now have observations with a valid lagged or lead variable but not with
    # actual valid state variables. Delete those by looking at the choice variable.
    merged_data = merged_data[merged_data["choice"].notna()]

    # We left too young people in the sample to construct lagged choice. Delete those
    # now.
    merged_data = merged_data[merged_data["age"] >= start_age]

    print(str(len(merged_data)) + " left after filtering missing lagged choices.")
    return merged_data


def create_policy_state(gebjahr):
    """This function creates the policy state according to the 2007 reform."""
    # Default state is 67
    policy_state = pd.Series(index=gebjahr.index, data=67, dtype=float)
    # Create masks for everyone born before 1964
    mask1 = (gebjahr <= 1964) & (gebjahr >= 1958)
    mask2 = (gebjahr <= 1958) & (gebjahr >= 1947)
    mask3 = gebjahr < 1947
    policy_state.loc[mask1] = 67 - 2 / 12 * (1964 - gebjahr[mask1])
    policy_state.loc[mask2] = 66 - 1 / 12 * (1958 - gebjahr[mask2])
    policy_state.loc[mask3] = 65
    return policy_state


def modify_policy_state(policy_states, options):
    """This function rounds policy state to the closest multiple of the policy
    expectations process grid size.

    min_SRA is set by the assumption of the belief process in the model.

    """
    min_SRA = options["min_SRA"]
    SRA_grid_size = options["SRA_grid_size"]
    policy_states = policy_states - min_SRA
    policy_id = np.around(policy_states / SRA_grid_size).astype(int)
    policy_states_values = min_SRA + policy_id * SRA_grid_size
    return policy_states_values, policy_id


def enforce_model_choice_restriction(merged_data, min_ret_age, max_ret_age):
    """This function filters the choice data according to the model setup.

    Specifically, it filters out people retire too early, work too long, or come back
    from retirement,

    """
    # Filter out people who are retired before min_ret_age
    merged_data = merged_data[
        ~((merged_data["choice"] == 2) & (merged_data["age"] < min_ret_age))
    ]
    merged_data = merged_data[
        ~((merged_data["lagged_choice"] == 2) & (merged_data["age"] <= min_ret_age))
    ]

    # Filter out people who are working after max_ret_age
    merged_data = merged_data[
        ~((merged_data["choice"] != 2) & (merged_data["age"] >= max_ret_age))
    ]
    # Filter out people who have not retirement as lagged choice after max_ret_age
    merged_data = merged_data[
        ~((merged_data["lagged_choice"] != 2) & (merged_data["age"] > max_ret_age))
    ]
    print(
        str(len(merged_data))
        + " left after dropping people who are retired before "
        + str(min_ret_age)
        + " or working after "
        + str(max_ret_age)
        + "."
    )

    # Filter out people who come back from retirement
    merged_data = merged_data[
        (merged_data["lagged_choice"] != 2) | (merged_data["choice"] == 2)
    ]

    print(
        str(len(merged_data))
        + " left after dropping people who come back from retirement."
    )
    return merged_data


def print_n_retirees(merged_data):
    n_retirees = merged_data.groupby("choice").size().loc[2]
    print(str(n_retirees) + " retirees in sample.")


def print_n_fresh_retirees(merged_data):
    n_fresh_retirees = (
        merged_data.groupby(["choice", "lagged_choice"]).size().loc[2, 1]
        + merged_data.groupby(["choice", "lagged_choice"]).size().loc[2, 0]
    )
    print(str(n_fresh_retirees) + " fresh retirees in sample.")


def print_sample_by_choice(merged_data, string):
    n_unemployed = merged_data.groupby(["choice"]).size().loc[0]
    n_workers = merged_data.groupby(["choice"]).size().loc[1]
    n_retirees = merged_data.groupby(["choice"]).size().loc[2]
    n_fresh_retirees = (
        merged_data.groupby(["choice", "lagged_choice"]).size().loc[2, 1]
        + merged_data.groupby(["choice", "lagged_choice"]).size().loc[2, 0]
    )
    print(
        "Left in sample:"
        + " unemployed: "
        + str(n_unemployed)
        + ", workers: "
        + str(n_workers)
        + ", retirees: "
        + str(n_retirees)
        + ", fresh retirees: "
        + str(n_fresh_retirees)
        + ", after"
        + string
    )
