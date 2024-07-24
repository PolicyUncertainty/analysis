import os

import numpy as np
import pandas as pd
from process_data.create_structural_est_sample import create_lagged_choice_variable
from process_data.create_structural_est_sample import filter_data
from process_data.soep_vars import create_choice_variable
from process_data.soep_vars import create_education_type
from process_data.soep_vars import generate_job_separation_var


def create_job_sep_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "job_sep_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Export parameters from specs
    start_year = specs["start_year"]
    end_year = specs["end_year"]
    start_age = specs["start_age"]
    max_ret_age = specs["max_ret_age"]

    # Load and merge data state data from SOEP core
    merged_data = load_and_merge_soep_core(paths["soep_c38"])

    # filter data (age, sex, estimation period)
    merged_data = filter_data(merged_data, start_year, end_year, start_age)

    # create choice and lagged choice variable
    merged_data = create_choice_variable(merged_data)
    # lagged choice
    merged_data = create_lagged_choice_variable(
        merged_data, start_year, end_year, start_age
    )

    # education
    merged_data = create_education_type(merged_data)

    # Job separation
    merged_data = generate_job_separation_var(merged_data)

    # Now restrict sample to all who worked last period or did loose their job
    merged_data = merged_data[
        (merged_data["lagged_choice"] == 1) | (merged_data["plb0282_h"] == 1)
    ]

    # Create age at which one got fired
    merged_data["age_fired"] = merged_data["age"] - 1

    merged_data.reset_index(drop=True, inplace=True)

    # Keep relevant columns
    merged_data = merged_data[
        [
            "age_fired",
            "education",
            "job_sep",
        ]
    ]
    merged_data = merged_data.astype(
        {
            "age_fired": np.int32,
            "education": np.int32,
            "job_sep": np.int32,
        }
    )
    # Rename age fired to age
    merged_data.rename(columns={"age_fired": "age"}, inplace=True)
    # Limit age range to start age and maximum retirement age
    merged_data = merged_data[
        (merged_data["age"] >= start_age) & (merged_data["age"] <= max_ret_age)
    ]
    print(f"{len(merged_data)} observations in job separation sample.")

    # save data
    merged_data.to_pickle(out_file_path)

    return merged_data


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
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data
