# %%
import os

import numpy as np
import pandas as pd

from process_data.auxiliary.filter_data import (
    drop_missings,
    filter_above_age,
    filter_below_age,
    filter_years,
    recode_sex,
)
from process_data.auxiliary.lagged_and_lead_vars import span_dataframe
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.health import correct_health_state, create_health_var


# %%
def create_health_transition_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = (
        paths["first_step_data"] + "health_transition_estimation_sample.pkl"
    )

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    df = load_and_merge_soep_core(paths["soep_c38"])

    # Pre-Filter estimation years
    df = filter_years(df, specs["start_year"] - 1, specs["end_year"] + 1)

    # Pre-Filter age and sex
    df = filter_below_age(df, specs["start_age"] - specs["health_smoothing_bandwidth"])
    df = filter_above_age(df, specs["end_age"] + specs["health_smoothing_bandwidth"])
    df = recode_sex(df)

    # Create education type
    df = create_education_type(df)

    # create health states
    df = create_health_var(df)
    df = span_dataframe(df, specs["start_year"] - 1, specs["end_year"] + 1)
    df = correct_health_state(df)

    out_cols = ["age", "education", "health", "lead_health", "sex"]

    df = drop_missings(df, out_cols)

    df = df[out_cols]

    print(
        str(len(df))
        + " observations in the final health transition sample.  \n ----------------"
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
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        f"{soep_c38_path}/pequiv.dta",
        columns=["pid", "syear", "m11126", "m11124"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="left"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="left")
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data
