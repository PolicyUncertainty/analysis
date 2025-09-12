import os

import numpy as np
import pandas as pd

from process_data.auxiliary.filter_data import (
    drop_missings,
    filter_below_age,
    filter_years,
    recode_sex,
)
from process_data.auxiliary.lagged_and_lead_vars import (
    span_dataframe,
)
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.health import correct_health_state, create_health_var
from process_data.soep_vars.job_hire_and_fire import generate_job_separation_var
from process_data.soep_vars.work_choices import create_choice_and_employment_status


def create_job_sep_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["first_step_data"] + "job_sep_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Load and merge data state data from SOEP core
    df = load_and_merge_soep_core(paths["soep_c38"])

    # We actually need the probablity to be fired for 1 year earlier,
    # to be able to integrate in the likelihood for unobserved 30-year olds.
    min_age_job_seps = specs["start_age"] - 1

    # Leave addtional age for leading
    df = filter_below_age(df, min_age_job_seps - 1)

    df = recode_sex(df)

    df = filter_years(df, specs["start_year"] - 1, specs["end_year"] + 1)

    # create choice and lagged choice variable
    df = create_choice_and_employment_status(df)
    # lagged choice
    df = span_dataframe(df, specs["start_year"] - 1, specs["end_year"] + 1)

    df["lagged_choice"] = df.groupby(["pid"])["choice"].shift()

    # We create the health variable and correct it
    df = create_health_var(df, filter_missings=False)
    df = correct_health_state(df)

    # Job separation
    df = generate_job_separation_var(df)
    # Overwrite job separation when individuals choose working
    df.loc[df["choice"] >= 2, "job_sep"] = 0

    # education
    df = create_education_type(df)

    # Now restrict sample to all who worked last period or did loose their job
    df = df[(df["lagged_choice"].isin([2, 3])) | (df["plb0282_h"] == 1)]
    # Kick out men that worked part-time last period
    df = df[~((df["lagged_choice"] == 2) & (df["sex"] == 0))]

    # Create age at which one got fired and rename job separation column
    df["age_fired"] = df["age"] - 1
    df.reset_index(inplace=True)

    # We also delete now the observations with invalid data, which we left before to have a continuous panel
    df = drop_missings(
        df=df,
        vars_to_check=["lagged_health", "lagged_choice", "age_fired", "education"],
    )

    # Relevant columns and datatype
    columns = {
        "age_fired": np.int32,
        "education": np.uint8,
        "lagged_health": np.uint8,
        "sex": np.uint8,
        "job_sep": np.uint8,
    }

    df = df[columns.keys()]
    df = df.astype(columns)
    # Rename age fired to age
    df.rename(columns={"age_fired": "age"}, inplace=True)
    # Limit age range to start age and maximum retirement age
    df = df[(df["age"] >= min_age_job_seps) & (df["age"] <= specs["max_ret_age"])]
    print(f"{len(df)} observations in job separation sample.")

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
            "pgstib",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )
    pequiv_data = pd.read_stata(
        # d11107: number of children in household
        # d11101: age of individual
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        f"{soep_c38_path}/pequiv.dta",
        columns=["pid", "syear", "d11107", "d11101", "m11126", "m11124"],
        convert_categoricals=False,
    )
    pathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["pid", "hid", "syear", "sex", "gebjahr"],
        convert_categoricals=False,
    )

    pl_data_reader = pd.read_stata(
        f"{soep_c38_path}/pl.dta",
        columns=["pid", "hid", "syear", "plb0304_h", "plb0282_h"],
        chunksize=100000,
        convert_categoricals=False,
    )
    pl_data = pd.DataFrame()

    for itm in pl_data_reader:
        pl_data = pd.concat([pl_data, itm])

    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        pgen_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    # Merge pgen data with pathl data and hl data
    merged_data = pd.merge(
        merged_data, pl_data, on=["pid", "hid", "syear"], how="inner"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="left")

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    del pgen_data, pathl_data
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data
