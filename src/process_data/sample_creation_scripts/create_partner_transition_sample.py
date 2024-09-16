# %%
import os

import numpy as np
import pandas as pd
from process_data.sample_creation_scripts.create_structural_est_sample import (
    filter_data,
)
from process_data.sample_creation_scripts.data_tools import filter_below_age
from process_data.sample_creation_scripts.data_tools import filter_by_sex
from process_data.sample_creation_scripts.data_tools import filter_est_years
from process_data.sample_creation_scripts.partner_code import (
    create_partner_and_lagged_state,
)
from process_data.var_resources.soep_vars import create_education_type


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

    start_year = specs["start_year"]
    end_year = specs["end_year"]
    start_age = specs["start_age"]

    df = load_and_merge_soep_core(paths["soep_c38"])

    df = create_education_type(df)

    # Filter estimation years
    df = filter_est_years(df, start_year, end_year)

    # The following code is dependent on span dataframe being called first.
    # In particular the lagged partner state must be after span dataframe and create partner state.
    # We should rewrite this
    df = span_dataframe(df, start_year, end_year)

    # In this function also merging is called
    df = create_partner_and_lagged_state(df)

    # Filter age and sex
    df = filter_below_age(df, start_age)
    df = filter_by_sex(df, no_women=False)

    df = keep_relevant_columns(df)
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


def span_dataframe(merged_data, start_year, end_year):
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
    full_container["hid"] = full_container.groupby(["pid"])["hid"].transform("last")
    return full_container


def keep_relevant_columns(df):
    df = df[
        ["age", "sex", "education", "partner_state", "lagged_partner_state", "children"]
    ]
    return df
