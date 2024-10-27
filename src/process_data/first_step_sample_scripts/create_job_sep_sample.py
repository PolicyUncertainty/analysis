import os

import numpy as np
import pandas as pd
from process_data.aux_scripts.filter_data import filter_data
from process_data.aux_scripts.lagged_and_lead_vars import (
    create_lagged_and_lead_variables,
)
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.job_hire_and_fire import generate_job_separation_var
from process_data.soep_vars.work_choices import create_choice_variable


def create_job_sep_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "job_sep_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Export parameters from specs
    start_age = specs["start_age"]
    max_ret_age = specs["max_ret_age"]

    # Load and merge data state data from SOEP core
    df = load_and_merge_soep_core(paths["soep_c38"])

    # filter data (age, sex, estimation period)
    df = filter_data(df, specs)

    # create choice and lagged choice variable
    df = create_choice_variable(df)

    # Job separation
    df = generate_job_separation_var(df)

    # lagged choice (Also leads the job separation variable, which we will not use.
    df = create_lagged_and_lead_variables(df, specs)

    # education
    df = create_education_type(df)

    # Now restrict sample to all who worked last period or did loose their job
    df = df[(df["lagged_choice"] == 1) | (df["plb0282_h"] == 1)]

    # Create age at which one got fired and rename job separation column
    df["age_fired"] = df["age"] - 1

    df.reset_index(drop=True, inplace=True)

    # Keep relevant columns
    df = df[
        [
            "age_fired",
            "education",
            "job_sep",
        ]
    ]
    df = df.astype(
        {
            "age_fired": np.int32,
            "education": np.int32,
            "job_sep": np.int32,
        }
    )
    # Rename age fired to age
    df.rename(columns={"age_fired": "age"}, inplace=True)
    # Limit age range to start age and maximum retirement age
    df = df[(df["age"] >= start_age) & (df["age"] <= max_ret_age)]
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

    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    del pgen_data, pathl_data
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data
