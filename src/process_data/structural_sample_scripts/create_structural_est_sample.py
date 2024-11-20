import os

import pandas as pd
from process_data.aux_scripts.filter_data import filter_data
from process_data.aux_scripts.lagged_and_lead_vars import (
    create_lagged_and_lead_variables,
)
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.experience import create_experience_variable
from process_data.soep_vars.job_hire_and_fire import determine_observed_job_offers
from process_data.soep_vars.job_hire_and_fire import generate_job_separation_var
from process_data.soep_vars.partner_code import create_partner_state
from process_data.soep_vars.wealth import add_wealth
from process_data.soep_vars.work_choices import create_choice_variable
from process_data.structural_sample_scripts.informed_state import create_informed_state
from process_data.structural_sample_scripts.model_restrictions import (
    enforce_model_choice_restriction,
)
from process_data.structural_sample_scripts.policy_state import create_policy_state


def create_structural_est_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "structural_estimation_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Load and merge data state data from SOEP core (all but wealth)
    df = load_and_merge_soep_core(soep_c38_path=paths["soep_c38"])

    # Create partner state(Merges also partners)
    df = create_partner_state(df, filter_missing=True)

    # (labor) choice
    df = create_choice_variable(df)

    # filter data. Leave additional years in for lagging and leading. For now no women
    df = filter_data(df, specs, no_women=True)

    # Job separation
    df = generate_job_separation_var(df)

    # lagged choice
    df = create_lagged_and_lead_variables(df, specs)

    # Add wealth data
    df = add_wealth(df, paths, specs)

    # Create period
    df["period"] = df["age"] - specs["start_age"]

    # policy_state
    df = create_policy_state(df, specs)

    # experience
    df = create_experience_variable(df)

    # education
    df = create_education_type(df)

    # additional restrictions based on model setup
    df = enforce_model_choice_restriction(df, specs)

    # Create informed state
    df = create_informed_state(df)

    # Construct job offer state
    was_fired_last_period = df["job_sep_this_year"] == 1
    df = determine_observed_job_offers(
        df, working_choices=[2, 3], was_fired_last_period=was_fired_last_period
    )

    # Keep relevant columns (i.e. state variables) and set their minimal datatype
    type_dict = {
        "period": "int8",
        "choice": "int8",
        "lagged_choice": "int8",
        "informed": "int8",
        "policy_state": "int8",
        "policy_state_value": "float32",
        "partner_state": "int8",
        "job_offer": "int8",
        "experience": "int8",
        "wealth": "float32",
        "education": "int8",
        "children": "int8",
        "full_observed_state": "bool",
    }
    df = df[list(type_dict.keys())]
    df = df.astype(type_dict)

    print_data_description(df)

    # Anonymize and save data
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(out_file_path)

    # save data
    df.to_pickle(out_file_path)

    return df


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
    pequiv_data = pd.read_stata(
        # d11107: number of children in household
        f"{soep_c38_path}/pequiv.dta",
        columns=["pid", "syear", "d11107"],
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data


def print_data_description(df):
    n_retirees = df.groupby("choice").size().loc[0]
    n_unemployed = df.groupby("choice").size().loc[1]
    n_part_time = df.groupby("choice").size().loc[2]
    n_full_time = df.groupby("choice").size().loc[3]
    n_fresh_retirees = (
        df.groupby(["choice", "lagged_choice"]).size().get((0, 1), 0)
        + df.groupby(["choice", "lagged_choice"]).size().get((0, 2), 0)
        + df.groupby(["choice", "lagged_choice"]).size().get((0, 3), 0)
    )
    print(str(len(df)) + " left in final estimation sample.")
    print("---------------------------")
    print(
        "Breakdown by choice:\n" + str(n_retirees) + " retirees [0] \n"
        "--"
        + str(n_fresh_retirees)
        + " thereof fresh retirees [0, lagged =!= 0] \n"
        + str(n_unemployed)
        + " unemployed [1] \n"
        + str(n_part_time)
        + " part-time [2] \n"
        + str(n_full_time)
        + " full time [3]."
    )
    print("---------------------------")
