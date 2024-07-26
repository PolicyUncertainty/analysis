import os

import numpy as np
import pandas as pd

from process_data.create_structural_est_sample import filter_data
from process_data.soep_vars import create_choice_variable_with_part_time

def create_partner_wage_est_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "wage_estimation_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    # Import parameters from specs
    start_year = specs["start_year"]
    end_year = specs["end_year"]
    start_age = specs["start_age"]


    df = load_and_merge_soep_core(paths["soep_c38"])
    df = filter_data(df, start_year, end_year, start_age, no_women=False)
    df = df[df["parid"] >= 0]
    print(str(len(df)) + " observations after dropping singles.")
    df = merge_couples(df)
    breakpoint()
    # create labor choice, keep only working full time and part time
    merged_data = create_choice_variable_with_part_time(merged_data)
    merged_data = merged_data[merged_data["choice"] == 1 | merged_data["choice"] == 3]



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

def merge_couples(df):
    """This function merges couples. It creates a '_p' variable for every column of df except syear, pid, hid, and parid."""
    df_partners = df.copy()
    merged_data = pd.merge(df, df_partners,how = 'inner' , left_on=["hid", "syear", "parid"] , right_on=["hid", "syear", "pid"], suffixes=("", "_p"))
    print(str(len(merged_data)) + " observations after merging couples.")
    # check if all pid-years are unique
    
    
    return merged_data
    
