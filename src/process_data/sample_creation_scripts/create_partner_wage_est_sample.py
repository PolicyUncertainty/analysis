import os

import pandas as pd
from process_data.sample_creation_scripts.create_partner_transition_sample import (
    create_partner_state,
)
from process_data.sample_creation_scripts.create_structural_est_sample import (
    filter_data,
)
from process_data.sample_creation_scripts.partner_code import merge_couples
from process_data.var_resources.soep_vars import create_choice_variable_with_part_time
from process_data.var_resources.soep_vars import create_education_type


def create_partner_wage_est_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "partner_wage_estimation_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    start_year = specs["start_year"]
    end_year = specs["end_year"]
    start_age = specs["start_age"]

    df = load_and_merge_soep_core(paths["soep_c38"])
    df = filter_data(df, start_year, end_year, start_age, no_women=False)
    df = wages_and_working_hours(df.copy())

    df = df[df["parid"] >= 0]
    print(str(len(df)) + " observations after dropping singles.")

    df = create_choice_variable_with_part_time(df)
    df = create_education_type(df)
    df = merge_couples(df)

    # Create partner state and drop if partner is absent or in non-working age
    df = create_partner_state(df, start_age)
    df = df[df["partner_state"] == 1]

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
            "pglabgro",
            "pgpsbil",
            "pgvebzeit",
        ],
        convert_categoricals=False,
    )
    pathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["pid", "hid", "parid", "syear", "sex", "gebjahr"],
        convert_categoricals=False,
    )

    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
    )

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    del pgen_data, pathl_data
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data


def wages_and_working_hours(df):
    df.rename(columns={"pglabgro": "wage"}, inplace=True)
    df.loc[df["wage"] < 0, "wage"] = 0
    print(str(len(df)) + " observations after dropping invalid wage values.")
    # working hours = contractual hours per week
    # df = df.rename(columns={"pgvebzeit": "working_hours"})
    # df = df[df["working_hours"] > 0]
    # print(str(len(df)) + " observations after dropping invalid working hours.")
    # df["hourly_wage"] = df["wage"] / (df["working_hours"] * (52 / 12))
    return df
