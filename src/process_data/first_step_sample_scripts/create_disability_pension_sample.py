# %%
import os

import numpy as np
import pandas as pd

from process_data.aux_and_plots.filter_data import (
    filter_above_age,
    filter_below_age,
    filter_years,
    recode_sex,
)
from process_data.aux_and_plots.lagged_and_lead_vars import span_dataframe
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.health import clean_health_create_states, create_health_var
from process_data.soep_vars.work_choices import create_choice_variable


# %%
def create_disability_pension_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = (
        paths["intermediate_data"] + "disability_pension_estimation_sample.pkl"
    )

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    df = load_and_merge_soep_core(paths["soep_c38"])

    df = recode_sex(df)

    # Create education type
    df = create_education_type(df)

    df = create_choice_variable(df)

    # create health states
    df = create_health_var(df)
    df = filter_years(df, specs["start_year"] - 1, specs["end_year"] + 1)

    df = span_dataframe(df, specs["start_year"] - 1, specs["end_year"] + 1)

    df["lagged_choice"] = df.groupby(["pid"])["choice"].shift()

    df = clean_health_create_states(df)

    # Drop missing lagged choices
    df = df[df["lagged_choice"].notna()]
    print(str(len(df)) + " left after filtering missing lagged choices.")

    # Now drop already retired people and people who are not in bad health
    df = df[df["lagged_choice"] != 0]
    df = df[df["health"] == 1]
    print(str(len(df)) + " left after filtering people who are not in bad health.")

    # Filter age and estimation years
    df = filter_above_age(df, specs["end_disability_age"])
    df = filter_below_age(df, specs["start_age"])
    df = filter_years(df, specs["start_year"], specs["end_year"])

    df["retirement"] = (df["choice"] == 0).astype(float)

    df = df[["age", "education", "sex", "retirement"]]

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
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data
