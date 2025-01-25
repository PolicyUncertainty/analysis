# %%
import os

import pandas as pd
from process_data.aux_and_plots.filter_data import filter_below_age
from process_data.aux_and_plots.filter_data import filter_years
from process_data.aux_and_plots.filter_data import recode_sex
from process_data.aux_and_plots.lagged_and_lead_vars import span_dataframe
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.partner_code import create_partner_state


# %%
def create_partner_transition_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = (
        paths["intermediate_data"] + "partner_transition_estimation_sample.pkl"
    )

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    df = load_and_merge_soep_core(paths["soep_c38"])

    df = create_education_type(df)

    # Filter estimation years
    df = filter_years(df, specs["start_year"], specs["end_year"])

    # In this function also merging is called
    df = create_partner_and_lagged_state(df, specs)

    # Filter age and sex
    df = filter_below_age(df, specs["start_age"])
    df = recode_sex(df)

    df = df[
        ["age", "sex", "education", "partner_state", "lead_partner_state", "children"]
    ]

    print(
        str(len(df))
        + " observations in the final partner transition sample.  \n ----------------"
    )
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
            "pgpsbil",
            "pgstib",
        ],
        convert_categoricals=False,
    )
    ppathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["syear", "pid", "hid", "sex", "parid", "gebjahr"],
        convert_categoricals=False,
    )
    pequiv_data = pd.read_stata(
        # d11107: number of children in household
        f"{soep_c38_path}/pequiv.dta",
        columns=["pid", "syear", "d11107"],
    )
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data


def create_partner_and_lagged_state(df, specs):
    # The following code is dependent on span dataframe being called first.
    # In particular the lagged partner state must be after span dataframe and create partner state.
    # We should rewrite this
    df = span_dataframe(df, specs["start_year"], specs["end_year"])

    df = create_partner_state(df)
    df["lead_partner_state"] = df.groupby(["pid"])["partner_state"].shift(-1)
    df = df[df["lead_partner_state"].notna()]
    df = df[df["partner_state"].notna()]
    print(
        str(len(df))
        + " observations after dropping people with a partner whose choice is not observed."
    )
    return df
