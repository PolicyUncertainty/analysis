import os

import pandas as pd
from process_data.aux_scripts.filter_data import filter_data
from process_data.aux_scripts.lagged_and_lead_vars import (
    create_lagged_and_lead_variables,
)
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.experience import create_experience_variable
from process_data.soep_vars.health import create_health_var
from process_data.soep_vars.job_hire_and_fire import determine_observed_job_offers
from process_data.soep_vars.job_hire_and_fire import generate_job_separation_var
from process_data.soep_vars.partner_code import create_partner_state
from process_data.soep_vars.wealth import add_wealth_impute_with_panel_reg
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

    df = create_partner_state(df, filter_missing=True)
    df = create_choice_variable(df)

    # filter data. Leave additional years in for lagging and leading. For now no women
    df = filter_data(df, specs, no_women=True)

    df = generate_job_separation_var(df)
    df = create_lagged_and_lead_variables(df, specs)
    df = add_wealth_impute_with_panel_reg(df, paths, specs)
    df["period"] = df["age"] - specs["start_age"]
    df = create_policy_state(df, specs)
    df = create_experience_variable(df)
    df = create_education_type(df)
    df = create_health_var(df)

    # enforce choice restrictions based on model setup
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
        "health_state": "int8",
    }
    df = df[list(type_dict.keys())]
    df = df.astype(type_dict)

    print_data_description(df)

    # Anonymize and save data
    df.reset_index(drop=True, inplace=True)
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

    # get household level data
    hl_data = pd.read_stata(
        f"{soep_c38_path}/hl.dta",
        columns=["hid", "syear", "hlc0043"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, hl_data, on=["hid", "syear"], how="left")
    pequiv_data = pd.read_stata(
        # d11107: number of children in household
        # d11101: age of individual
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        f"{soep_c38_path}/pequiv.dta",
        columns=["pid", "syear", "d11107", "d11101", "m11126", "m11124"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)

    merged_data["age"] = merged_data["d11101"].astype(int)
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
