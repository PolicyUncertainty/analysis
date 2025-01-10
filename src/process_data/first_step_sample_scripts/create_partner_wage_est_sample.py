import os

import pandas as pd
from process_data.aux_scripts.filter_data import filter_data
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.partner_code import create_partner_state


def create_partner_wage_est_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "partner_wage_estimation_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    df = load_and_merge_soep_core(paths["soep_c38"])
    df = filter_data(df, specs)
    df = create_wages(df.copy())

    # Drop singles
    df = df[df["parid"] >= 0]
    print(str(len(df)) + " observations after dropping singles.")

    df = create_education_type(df)

    # Create partner state and drop if partner is absent or in non-working age
    df = create_partner_state(df)
    df = df[df["partner_state"] == 1]

    df.reset_index(inplace=True)
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
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data


def create_wages(df):
    df.rename(columns={"pglabgro": "wage"}, inplace=True)
    df.loc[df["wage"] < 0, "wage"] = 0
    print(str(len(df)) + " observations after dropping invalid wage values.")
    return df
