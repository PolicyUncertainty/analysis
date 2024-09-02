#%%
import os
import numpy as np
import pandas as pd

from process_data.create_structural_est_sample import filter_data
from process_data.soep_vars import create_education_type
from process_data.soep_vars import create_choice_variable_with_part_time
from process_data.create_partner_wage_est_sample import merge_couples

#%%
def create_partner_transition_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "partner_transition_estimation_sample.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    start_year = specs["start_year"]
    end_year = specs["end_year"]
    start_age = specs["start_age"]

    df = load_and_merge_soep_core(paths["soep_c38"])
    df = filter_data(df, start_year, end_year, start_age, no_women=False)
    df = create_education_type(df)
    df = create_choice_variable_with_part_time(df)
    df = span_dataframe(df, start_year, end_year)
    df = merge_couples(df, keep_singles = True)
    df = create_partner_state(df, start_age)
    df = keep_relevant_columns(df)
    print(str(len(df)) + " observations in the final partner transition sample.")
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
        columns=[
            "syear",
            "pid",
            "hid",
            "sex",
            "parid",
            "gebjahr"
        ],
        convert_categoricals=False,
    )
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    return merged_data

def create_partner_state(df, start_age):
    '''0: no partner, 1: working-age partner, 2: retired partner'''
    # has to be done for both state and lagged state
    # people with a partner whose choice is not observed stay in this category
    df.loc[:, "partner_state"] = np.nan
    # no partner (no parid)
    df.loc[:, "partner_state"] = np.where(df["parid"] < 0, 0, df["partner_state"])
    # working-age partner (choice 0, 1, 3)
    df.loc[:, "partner_state"] = np.where(df["choice_p"] == 0, 1, df["partner_state"])
    df.loc[:, "partner_state"] = np.where(df["choice_p"] == 1, 1, df["partner_state"])
    df.loc[:, "partner_state"] = np.where(df["choice_p"] == 3, 1, df["partner_state"])
    # retired partner (choice 2)
    df.loc[:, "partner_state"] = np.where(df["choice_p"] == 2, 2, df["partner_state"])
    # drop nans
    # df = df[df[string + "partner_state"].notna()]
    df["lagged_partner_state"] = df.groupby(["pid"])["partner_state"].shift()
    df = df[df["lagged_partner_state"].notna()]
    df = df[df["partner_state"].notna()]
    # We left people who are too young in the sample to construct lagged choice. Delete those now.
    df = df[df["age"] >= start_age]
    print(str(len(df)) + " observations after dropping people with a partner whose choice is not observed.")
    return df

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
    df= df[["age", "sex", "education", "partner_state","lagged_partner_state"]]
    return df



