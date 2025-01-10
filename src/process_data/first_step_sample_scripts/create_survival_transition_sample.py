# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from process_data.aux_scripts.filter_data import filter_above_age
from process_data.aux_scripts.filter_data import filter_below_age
from process_data.aux_scripts.filter_data import filter_years
from process_data.aux_scripts.filter_data import recode_sex
from process_data.aux_scripts.lagged_and_lead_vars import span_dataframe
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.health import clean_health_create_states
from process_data.soep_vars.health import create_health_var


# %%
def create_survival_transition_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    # cleaned sample
    out_file_path = (
        paths["intermediate_data"] + "mortality_transition_estimation_sample.pkl"
    )

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    df = load_and_merge_datasets(paths["soep_c38"], specs)

    # filter age and estimation years
    df = filter_below_age(df, specs["start_age_mortality"])
    df = filter_above_age(df, specs["end_age_mortality"])
    df = filter_years(df, specs["start_year_mortality"], specs["end_year_mortality"])

    # create columns for the start age and health state
    df = create_start_age_and_health(df)

    df = df[
        [
            "age",
            "start_age",
            "event_death",
            "education",
            "sex",
            "health",
            "start_health",
        ]
    ]

    # set the dtype of the columns to float
    df = df.astype(float)

    # sum the death events for the entire sample
    print(
        "Death events in the sample: ",
        f"{len(df[df['event_death'] == 1])} (total) = "
        f"{len(df[(df['event_death'] == 1) & (df['health'] == 1)])} (health 1) + "
        f"{len(df[(df['event_death'] == 1) & (df['health'] == 0)])} (health 0)",
    )

    print(
        f"Years: {df.index.get_level_values('syear').min()}-{df.index.get_level_values('syear').max()}, "
        f"Min age: {df['age'].min()}, Max age: {df['age'].max()}, Avg age: {round(df['age'].mean(), 2)}, "
        f"Unique individuals: {df.index.get_level_values('pid').nunique()}, "
        f"Avg time in sample: {round(df.groupby('pid').size().mean(), 2)}"
    )

    print(
        str(len(df))
        + " observations in the final survival transition sample.  \n ----------------"
    )

    df.to_pickle(out_file_path)

    # duplicate the sample - Kroll and Lampert (2009)
    out_file_path_dupli = (
        paths["intermediate_data"]
        + "mortality_transition_estimation_sample_duplicated.pkl"
    )

    # Create original and duplicated samples
    df1 = df.copy().reset_index()
    df2 = df.copy().reset_index()

    # Modify df2 with unknown values
    df2["education"] = np.nan
    df2["health"] = np.nan
    df2["start_health"] = np.nan

    # Add true_sample indicators
    df1["true_sample"] = 1
    df2["true_sample"] = 0

    # Concatenate the two samples
    dupli_df = pd.concat([df1, df2])
    dupli_df.set_index(["pid", "syear", "true_sample"], inplace=True)
    dupli_df.sort_index(inplace=True)

    # Create interaction indicators for health and education
    dupli_df = create_interaction_columns(
        dupli_df, ("health", "health"), ("education", "edu")
    )

    # Create interaction indicators for start health and education
    dupli_df = create_interaction_columns(
        dupli_df, ("start_health", "start_health"), ("education", "edu")
    )

    # Convert DataFrame to floats for computation
    dupli_df = dupli_df.astype(float)

    # Save the duplicated sample
    dupli_df.to_pickle(out_file_path_dupli)

    # return the true sample
    return df


def load_and_merge_datasets(soep_c38_path, specs):
    time_invariant_data = load_and_process_soep_yearly_survey_data(soep_c38_path, specs)
    life_spell_data = load_and_process_life_spell_data(soep_c38_path, specs)
    health_data = load_and_process_soep_health(soep_c38_path, specs)

    # merge time invariant data onto the life spell data
    df = pd.merge(life_spell_data, time_invariant_data, on="pid", how="left")
    # combine the life spell data with the health data (keep intersection)
    df = pd.merge(df, health_data, on=["pid", "syear"], how="inner")

    # set index and age
    df["age"] = df["syear"] - df["gebjahr"]
    df = df.set_index(["pid", "syear"])

    return df


def load_and_process_soep_yearly_survey_data(soep_c38_path, specs):
    """Load the annual data from the SOEP C38 dataset and process it."""
    # Load SOEP core data
    pgen_data = pd.read_stata(
        f"{soep_c38_path}/pgen.dta",
        columns=[
            "syear",
            "pid",
            "pgpsbil",
        ],
        convert_categoricals=False,
    )
    ppathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["syear", "pid", "sex", "gebjahr"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(pgen_data, ppathl_data, on=["pid", "syear"], how="inner")
    merged_data.set_index(["pid", "syear"], inplace=True)

    # keep male and female obs. and transform to 0/1 = male/female
    df = recode_sex(merged_data)
    # create education type variable
    df = create_education_type(df)

    # keep only the one observation for each individual
    # with the highest education level + invariant variables
    df = df.groupby("pid")[["sex", "education", "gebjahr"]].max()
    return df


def load_and_process_life_spell_data(soep_c38_path, specs):
    lifespell_data = pd.read_stata(
        f"{soep_c38_path}/lifespell.dta",
        convert_categoricals=False,
    )
    # --- Generate spell duration and expand dataset --- lifespell data
    lifespell_data["spellduration"] = (
        lifespell_data["end"] - lifespell_data["begin"]
    ) + 1
    lifespell_data_long = lifespell_data.loc[
        lifespell_data.index.repeat(lifespell_data["spellduration"])
    ].reset_index(drop=True)
    # --- Generate syear --- lifespell data
    lifespell_data_long["n"] = (
        lifespell_data_long.groupby(["pid", "spellnr"]).cumcount() + 1
    )  # +1 since cumcount starts at 0
    lifespell_data_long["syear"] = (
        lifespell_data_long["begin"] + lifespell_data_long["n"] - 1
    )
    # --- Keep only relevant columns --- lifespell data
    columns_to_keep = ["pid", "syear", "spelltyp", "spellnr"]
    lifespell_data_long = lifespell_data_long[columns_to_keep]
    # --- Generate death event variable --- lifespell data
    lifespell_data_long["event_death"] = (lifespell_data_long["spelltyp"] == 4).astype(
        "int"
    )

    # Split into dataframes of death and not death
    not_death_idx = lifespell_data_long[lifespell_data_long["event_death"] == 0].index
    first_death_idx = (
        lifespell_data_long[lifespell_data_long["event_death"] == 1]
        .groupby("pid")["syear"]
        .idxmin()
    )

    # Final index and df
    final_index = not_death_idx.union(first_death_idx)
    lifespell_data_long = lifespell_data_long.loc[final_index]

    return lifespell_data_long


def load_and_process_soep_health(soep_c38_path, specs):
    pequiv_data = pd.read_stata(
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual
        f"{soep_c38_path}/pequiv.dta",
        columns=["pid", "syear", "m11126", "m11124"],
        convert_categoricals=False,
    )
    pequiv_data.set_index(["pid", "syear"], inplace=True)
    pequiv_data.sort_index(inplace=True)

    # create health state variable and span the dataframe
    pequiv_data = create_health_var(pequiv_data)
    pequiv_data = span_dataframe(
        pequiv_data, specs["start_year_mortality"], specs["end_year_mortality"]
    )
    pequiv_data = clean_health_create_states(pequiv_data)
    # Fill health gaps
    pequiv_data = fill_health_gaps_vectorized(pequiv_data)
    # forward fill health state for every individual
    pequiv_data["health"] = pequiv_data.groupby("pid")[
        "health"
    ].ffill()  # TO DO: this makes the fill gaps function obsolete and is a very strong assumption

    # # for deaths, set the health state to the last known health state
    # pequiv_data["last_known_health"] = pequiv_data.groupby("pid")["health"].transform("last")
    # pequiv_data.loc[
    #     (pequiv_data["event_death"] == 1) & (pequiv_data["health"].isna()), "health"
    # ] = pequiv_data.loc[
    #     (pequiv_data["event_death"] == 1) & (pequiv_data["health"].isna()),
    #     "last_known_health",
    # ]

    # drop individuals without any health state information
    pequiv_data = pequiv_data[(pequiv_data["health"].notna())]

    return pequiv_data


def fill_health_gaps_vectorized(df):
    """Fill gaps where the first and last known health state are identical.

    Parameters:
        df (DataFrame): The DataFrame containing the "health" column.

    Returns:
        DataFrame: The modified DataFrame with filled health state gaps.

    """
    ffilled = df.groupby("pid")["health"].ffill()
    bfilled = df.groupby("pid")["health"].bfill()
    agreeing_mask = ffilled == bfilled
    df["health"] = np.where(df["health"].isna() & agreeing_mask, ffilled, df["health"])
    return df


def create_start_age_and_health(df):
    """Determine the starting age and health state for each "pid".

    Parameters:
        df (DataFrame): The DataFrame containing "pid", "age", and "syear" columns.

    Returns:
        DataFrame: The modified DataFrame with "start_age" and "start_health" columns.

    """
    df = df.reset_index()
    df["start_age"] = df.groupby("pid")["age"].transform("min")
    indx = df.groupby("pid")["syear"].idxmin()
    df["start_health"] = np.nan
    df.loc[indx, "start_health"] = df.loc[indx, "health"]
    df["start_health"] = df.groupby("pid")["start_health"].transform("max")
    df = df.set_index(["pid", "syear"])
    return df


def create_interaction_columns(df, col1_info, col2_info):
    """Create interaction indicator columns based on two 0-1-columns in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame to modify.
        col1_info (tuple): A tuple containing the column name and prefix for the first column (e.g., ("col1", "prefix1")).
        col2_info (tuple): A tuple containing the column name and prefix for the second column (e.g., ("col2", "prefix2")).

    Returns:
        DataFrame: The modified DataFrame with new interaction columns.

    """
    column1, prefix1 = col1_info
    column2, prefix2 = col2_info
    combinations = [(1, 1), (1, 0), (0, 1), (0, 0)]
    for val1, val2 in combinations:
        col_name = f"{prefix1}{val1}_{prefix2}{val2}"
        df[col_name] = (df[column1] == val1) & (df[column2] == val2)
    return df
