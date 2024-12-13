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
    # df = filter_years(df, specs["start_year_mortality"], specs["end_year_mortality"])

    # Fill health states
    # IF: The current value of health is missing (mi(health[_n])).
    # The value x observations ahead (health[_n+x']) is not missing (!mi(health[_n+x'])). AND
    # The value immediately before the current observation (health[_n-1]) is not missing (!mi(health[_n-1])). AND
    # The values immediately before the current observation (health[_n-1]) and x observations ahead (health[_n+x']) are equal (health[_n-1] == health[_n+x']).
    # THEN: the missing value of health at the current position (_n) is replaced by the value of health[_n+x']`.

    # fill gaps were first and last known health state are identical with that value
    def fill_health_gaps(group):
        # Forward-fill and backward-fill
        ffilled = group["health_state"].ffill()
        bfilled = group["health_state"].bfill()
        # Create a mask where forward-fill and backward-fill agree
        agreeing_mask = ffilled == bfilled
        # Fill only where the mask is True
        group["health_state"] = group["health_state"].where(
            ~group["health_state"].isna() | ~agreeing_mask, ffilled
        )
        return group

    # Fill health gaps
    df = df.sort_index()
    df = df.groupby("pid").apply(fill_health_gaps)
    df.index = df.index.droplevel(
        1
    )  # remove extra pid index level -> index = (pid, syear) instead of (pid, pid, syear)

    print("Obs. after filling health gaps:", len(df))

    # print number of death events for the entire sample with and without missing health data
    print("Death events in the sample: ", df["event_death"].sum())
    print(
        "Death events in the sample without missing health data: ",
        df[df["health_state"].notna()]["event_death"].sum(),
    )
    print(
        "Number of observations in the sample with health 1 in the death year:",
        len(df[(df["event_death"] == 1) & (df["health_state"] == 1)]),
    )
    print(
        "Number of observations in the sample with health 0 in the death year:",
        len(df[(df["event_death"] == 1) & (df["health_state"] == 0)]),
    )

    # for deaths, set the health state to the last known health state
    df["last_known_health_state"] = df.groupby("pid")["health_state"].transform("last")
    df.loc[
        (df["event_death"] == 1) & (df["health_state"].isna()), "health_state"
    ] = df.loc[
        (df["event_death"] == 1) & (df["health_state"].isna()),
        "last_known_health_state",
    ]
    print("Setting health state to last known health state for deaths.")

    # print number of death events for the entire sample with and without missing health data
    print("Death events in the sample: ", df["event_death"].sum())
    print(
        "Death events in the sample without missing health data: ",
        df[df["health_state"].notna()]["event_death"].sum(),
    )
    print(
        "Number of observations in the sample with health 1 in the death year:",
        len(df[(df["event_death"] == 1) & (df["health_state"] == 1)]),
    )
    print(
        "Number of observations in the sample with health 0 in the death year:",
        len(df[(df["event_death"] == 1) & (df["health_state"] == 0)]),
    )

    # drop if the health state is missing
    df = df[(df["health_state"].notna())]
    print("Obs. after dropping missing health data:", len(df))

    # for every pid find first year in the sample observations and set the begin age and health state
    df = df.reset_index()
    df["begin_age"] = df.groupby("pid")["age"].transform("min")
    indx = df.groupby("pid")["syear"].idxmin()
    df["begin_health_state"] = 0
    df.loc[indx, "begin_health_state"] = df.loc[indx, "health_state"]
    df["begin_health_state"] = df.groupby("pid")["begin_health_state"].transform("max")
    df = df.set_index(["pid", "syear"])

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

    # Show data overview
    print(df.head())
    # sum the death events for the entire sample
    print("Death events in the sample:")
    print(df["event_death"].sum())

    # print the min and max age in the sample
    print("Min age in the sample:", df["age"].min())
    print("Max age in the sample:", df["age"].max())

    # print the average age in the sample
    print("Average age in the sample:", round(df["age"].mean(), 2))

    # print the number of unique individuals in the sample
    print(
        "Number of unique individuals in the sample:",
        df.index.get_level_values("pid").nunique(),
    )

    # print the number of unique years in the sample (min and max)
    print(
        "Sample Years:",
        df.index.get_level_values("syear").min(),
        "-",
        df.index.get_level_values("syear").max(),
    )

    # Average time spent in the sample for each individual
    print(
        "Average time spent in the sample for each individual:",
        round(df.groupby("pid").size().mean(), 2),
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

    # Pre-Filter estimation years
    df = filter_years(
        merged_data, specs["start_year_mortality"], specs["end_year_mortality"]
    )
    df = filter_by_sex(df, no_women=False)
    # Create education type
    df = create_education_type(df)

    full_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [df.index.get_level_values("pid").unique(), range(1992, 2017)],
            names=["pid", "syear"],
        ),
        columns=["sex", "education", "gebjahr"],
    )
    full_df.update(df)
    full_df["education"] = full_df.groupby("pid")["education"].transform("max")
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
    ).drop(
        [
            "zensor",
            "info",
            "study1992",
            "study2001",
            "study2006",
            "study2008",
            "flag1",
            "immiyearinfo",
        ],
        axis=1,
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
    # --- Clean-up --- lifespell data
    lifespell_data_long = lifespell_data_long[
        lifespell_data_long["syear"] <= specs["end_year"] + 1
    ]
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

    # create health states
    pequiv_data = filter_years(
        pequiv_data, specs["start_year_mortality"] - 1, specs["end_year_mortality"] + 1
    )
    pequiv_data = create_health_var(pequiv_data)
    pequiv_data = span_dataframe(
        pequiv_data, specs["start_year_mortality"] - 1, specs["end_year_mortality"] + 1
    )
    pequiv_data = clean_health_create_states(pequiv_data)

    return pequiv_data
