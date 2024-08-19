import os

import numpy as np
import pandas as pd

from process_data.create_structural_est_sample import filter_data
from process_data.soep_vars import create_education_type
from process_data.soep_vars import create_choice_variable_with_part_time

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
    df = wages_and_working_hours(df)
       
    df = df[df["parid"] >= 0]
    print(str(len(df)) + " observations after dropping singles.")

    df = create_choice_variable_with_part_time(df)
    df = df[(df["choice"] == 1) | (df["choice"] == 3)] 
    print(str(len(df)) + " observations after unemployed and retirees.")

    df = create_education_type(df)
    df = merge_couples(df) #partner data called {var}_p 
    df = keep_relevant_columns(df)
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
    df = df[df["wage"] > 0]
    print(str(len(df)) + " observations after dropping invalid wage values.")
    # working hours = contractual hours per week 
    df.rename(columns={"pgvebzeit": "working_hours"}, inplace=True)
    df = df[df["working_hours"] > 0]
    print(str(len(df)) + " observations after dropping invalid working hours.")
    df["hourly_wage"] = df["wage"] / (df["working_hours"] * (52 / 12))
    return df

def merge_couples(df):
    """This function merges couples based on the 'parid' identifier.
    Partner variables are market '_p' in the merged dataset."""
    df = df.reset_index()
    df_partners = df.copy()
    merged_data = pd.merge(df, df_partners,how = 'inner' , left_on=["hid", "syear", "parid"] , right_on=["hid", "syear", "pid"], suffixes=("", "_p"))
    print(str(len(merged_data)) + " observations after merging couples.")
    return merged_data
    
def keep_relevant_columns(df):
    df = df[
        [
            "pid",
            "parid",
            "age",
            "wage",
            "education",
            "syear",
            "choice",
            "sex",
            "hourly_wage",
            "age_p",
            "wage_p",
            "education_p",
            "choice_p",
            "sex_p",
            "hourly_wage_p"
        ]
    ]
    df = df.astype(
        {
            "pid": np.int32,
            "parid": np.int32,
            "syear": np.int32,
            "age": np.int32,
            "wage": np.float64,
            "education": np.int32,
            "choice": np.int32,
            "sex": np.int32,
            "hourly_wage": np.float64,
            "age_p": np.int32,
            "wage_p": np.float64,
            "education_p": np.int32,
            "choice_p": np.int32,
            "sex_p": np.int32,
            "hourly_wage_p": np.float64,
        }
    )
    print(str(len(df)) + " observations in final wage estimation dataset.")
    return df