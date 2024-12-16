# %%
import os

import numpy as np
import pandas as pd
from process_data.aux_scripts.filter_data import filter_above_age
from process_data.aux_scripts.filter_data import filter_below_age
from process_data.aux_scripts.filter_data import filter_by_sex
from process_data.aux_scripts.filter_data import filter_years
from process_data.aux_scripts.lagged_and_lead_vars import span_dataframe
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.health import clean_health_create_states
from process_data.soep_vars.health import create_health_var


# %%
def create_survival_transition_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = (
        paths["intermediate_data"] + "mortality_transition_estimation_sample.pkl"
    )

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    df = load_and_merge_datasets(paths["soep_c38"], specs)

    # filter age and estimation years
    df = filter_below_age(df, 16)
    df = filter_above_age(df, 110)

    # create columns for the beginning age and health state
    df = create_begin_age_and_health_state(df)

    df = df[
        [
            "age",
            "begin_age",
            "event_death",
            "education",
            "sex",
            "health_state",
            "begin_health_state",
        ]
    ]

    # set the dtype of the columns to float
    df = df.astype(float)


    # sum the death events for the entire sample
    print("Death events in the sample: ", 
      f"{len(df[df['event_death'] == 1])} (total) = "
      f"{len(df[(df['event_death'] == 1) & (df['health_state'] == 1)])} (health 1) + "
      f"{len(df[(df['event_death'] == 1) & (df['health_state'] == 0)])} (health 0)"
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
    return df


def load_and_merge_datasets(soep_c38_path, specs):
    annual_survey_data = load_and_process_soep_yearly_survey_data(soep_c38_path, specs)
    life_spell_data = load_and_process_life_spell_data(soep_c38_path, specs)
    health_data = load_and_process_soep_health(soep_c38_path, specs)
    df = pd.merge(annual_survey_data, life_spell_data, on=["pid", "syear"], how="inner")
    df = pd.merge(df, health_data, on=["pid", "syear"], how="inner")
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

    df = filter_by_sex(merged_data, no_women=False) # keep male and female obs.
    df = create_education_type(merged_data) # create education type

    full_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [df.index.get_level_values("pid").unique(), range(1992, 2017)],
            names=["pid", "syear"],
        ),
        columns=["sex", "education", "gebjahr"],
    )
    full_df.update(df)
    full_df["education"] = full_df.groupby("pid")["education"].transform("max") # education is constant for each individual (max value)
    full_df["sex"] = full_df.groupby("pid")["sex"].transform("max")
    full_df["gebjahr"] = full_df.groupby("pid")["gebjahr"].transform("max")
    full_df["age"] = full_df.index.get_level_values("syear") - full_df["gebjahr"]
    full_df.drop("gebjahr", axis=1, inplace=True)

    # Pre-Filter age and sex
    full_df = filter_below_age(full_df, 16)
    full_df = filter_above_age(full_df, 110)

    return full_df


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
    columns_to_keep = ["pid", "syear", "spellnr"]
    lifespell_data_long = lifespell_data_long[columns_to_keep]
    # --- Generate death event variable --- lifespell data
    lifespell_data_long["event_death"] = (lifespell_data_long["spellnr"] == 4).astype(
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

    # create health states
    min_syear = pequiv_data.index.get_level_values("syear").min()
    max_syear = pequiv_data.index.get_level_values("syear").max()
    pequiv_data = create_health_var(pequiv_data)
    pequiv_data = span_dataframe(pequiv_data, min_syear, max_syear)
    pequiv_data = clean_health_create_states(pequiv_data)

    # Fill health gaps 
    pequiv_data = fill_health_gaps_vectorized(pequiv_data)

    # # for deaths, set the health state to the last known health state
    # pequiv_data["last_known_health_state"] = pequiv_data.groupby("pid")["health_state"].transform("last")
    # pequiv_data.loc[
    #     (pequiv_data["event_death"] == 1) & (pequiv_data["health_state"].isna()), "health_state"
    # ] = pequiv_data.loc[
    #     (pequiv_data["event_death"] == 1) & (pequiv_data["health_state"].isna()),
    #     "last_known_health_state",
    # ]

    # forward fill health state for every individual 
    # TO DO: this makes the fill gaps function obsolete and is a very strong assumption
    pequiv_data["health_state"] = pequiv_data.groupby("pid")["health_state"].ffill() 

    # drop individuals without any health state information 
    pequiv_data = pequiv_data[(pequiv_data["health_state"].notna())] 

    return pequiv_data


# fill gaps were first and last known health state are identical with that value
def fill_health_gaps_vectorized(df):
    ffilled = df.groupby("pid")["health_state"].ffill()
    bfilled = df.groupby("pid")["health_state"].bfill()
    agreeing_mask = ffilled == bfilled
    df["health_state"] = np.where(df["health_state"].isna() & agreeing_mask, ffilled, df["health_state"])
    return df

# for every pid find the first observation year in the sample observations and save its age and health state
def create_begin_age_and_health_state(df):
    df = df.reset_index()
    df["begin_age"] = df.groupby("pid")["age"].transform("min")
    indx = df.groupby("pid")["syear"].idxmin()
    df["begin_health_state"] = np.nan
    df.loc[indx, "begin_health_state"] = df.loc[indx, "health_state"]
    df["begin_health_state"] = df.groupby("pid")["begin_health_state"].transform("max")
    df = df.set_index(["pid", "syear"])
    return df