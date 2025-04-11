import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_data.soep_vars.artkalen import prepare_artkalen_data
from process_data.soep_vars.birth import create_float_birth_year
from process_data.soep_vars.interview_date import create_float_interview_date
from process_data.soep_vars.work_choices import create_choice_variable


def create_choice_variable_from_artkalen(
    path_dict, specs, df, load_artkalen_choice=True
):

    relevant_pids = df.index.get_level_values("pid").unique().tolist()
    artkalen_data = prepare_artkalen_data(
        path_dict, relevant_pids, specs["start_year"] - 1, specs["end_year"] + 1
    )

    df = create_float_interview_date(df)
    df = create_float_birth_year(df)

    if not load_artkalen_choice:
        df["choice"] = np.nan
        df["float_age"] = df["age"].astype(float)
        # With create artkalen choice
        partial_select = lambda pid_group: select_spell_for_pid(
            pid_group, artkalen_data
        )
        df = df.groupby("pid").apply(partial_select)
        df["art_choice"] = df["choice"].copy()
        df[["art_choice", "float_age"]].to_pickle(
            path_dict["intermediate_data"] + "art_choice.pkl"
        )
    else:
        df[["art_choice", "float_age"]] = pd.read_pickle(
            path_dict["intermediate_data"] + "art_choice.pkl"
        )

    df["lagged_art_choice"] = df.groupby("pid")["art_choice"].shift(1)

    # Create pgen choice and overwrite
    df = create_choice_variable(df, filter_missings=False)
    df["pgen_choice"] = df["choice"].copy()

    df["choice"] = df["art_choice"].copy()
    nan_mask = df["choice"].isna()
    cont_choice = df["pgen_choice"] == df["lagged_art_choice"]
    df.loc[nan_mask & cont_choice, "choice"] = df.loc[
        nan_mask & cont_choice, "pgen_choice"
    ]

    df["lagged_choice"] = df.groupby("pid")["choice"].shift(1)
    return df


def select_spell_for_pid(pid_group, artkalen_data):
    """
    Select the spells for a given pid group.
    """
    pid = pid_group.index.get_level_values("pid")[0]

    # We first need to clean the spells and assign a correct choice
    # Select pid spells of arkalen
    pid_spells = artkalen_data[artkalen_data["pid"] == pid].sort_values("begin")

    if len(pid_spells) == 0:
        return pid_group.loc[(pid, (slice(None)))]

    already_retired = False
    id_first_retirement = -1
    past_ret_ids = []
    for interview_count in range(len(pid_group)):
        id = pid_group.index[interview_count]
        interview_date = pid_group.loc[id, "float_interview"]
        if np.isnan(interview_date):
            # Take syear (second part of id) and add 0.5. This observation will be dropped later,
            # but it might be usefull for lagging or leading
            interview_date = id[1] + 0.5

        overlapping_spells = pid_spells[
            (pid_spells["float_begin"] <= interview_date)
            & (pid_spells["float_end"] >= interview_date)
        ]
        choice = select_dominant_spelltyp_and_recode(
            overlapping_spells, pid_group.loc[id, "pgemplst"]
        )
        pid_group.loc[id, "choice"] = choice

        # Now lets treat retirement
        ret_choice = choice == 0

        # If retirement is chosen we add the id to the list of past retirements
        if ret_choice:
            past_ret_ids += [id]

        # If this is the first time, we change the age of the observation to the float age
        if ret_choice & (not already_retired):
            start_first_ret_spell = overlapping_spells[
                overlapping_spells["spelltyp"] == 6
            ].iloc[0]["float_begin"]
            pid_group.loc[id, "float_age"] = (
                start_first_ret_spell - pid_group.iloc[0]["float_birth_year"]
            )
            id_first_retirement = id
            already_retired = True

        # If a person is already retired, but now changes back, we need to set all past retirement choices
        # to unemployment, set already retired to False and set the float age back to the even age in the first
        # retirement observation
        if (not ret_choice) & already_retired & ~np.isnan(choice):
            for past_id in past_ret_ids:
                pid_group.loc[past_id, "choice"] = 1
            pid_group.loc[id_first_retirement, "float_age"] = pid_group.loc[
                id_first_retirement, "age"
            ]
            already_retired = False
            past_ret_ids = []

    return pid_group.loc[(pid, (slice(None)))]


def select_dominant_spelltyp_and_recode(spells, employment_status):
    """Select choice by spelltyp and recode it to the choice variable.

    This functions implements the following rules. They are exclusive downwards. So we always select
    a rule further downwards, if none was full-filled before:

        - If there are no spells, we return nan
        - If there is an education spell (8) or a training spell (4, 13), which constitutes a
          initial education, we classify them as nan, as they will be dropped later. Also if they
          are in military service (Zivil/Wehrdienst)
        - If there is full-time employment (1) or short hours (2), we select this as full-time
        - If there is retirement (6) we select this.
        - If there is regular part-time (3) then we select this.
          We do not select minijobs (15) as part-time. They will be dropped later
        - If there is unemployment (5), parental leave (7), housewife-husband (10) or retraining (14),
          we select this as unemployment.
        - We drop the observation if there is non of the above, but side job (11), "other" (12),
          marginal employment (15) or missing (99)

    We code:
        - 0: retirement
        - 1: unemployment
        - 2: part-time
        - 3: full-time

    """
    if len(spells) == 0:
        return np.nan
    # Education spells
    elif spells["spelltyp"].isin([4, 8, 9, 13]).any():
        return np.nan
    # Full-time spells
    elif spells["spelltyp"].isin([1, 2]).any():
        return 3
    # Retirement spells
    elif spells["spelltyp"].isin([6]).any():
        return 0
    # Part-time spells
    elif spells["spelltyp"].isin([3]).any():
        return 2
    # Unemployment spells
    elif spells["spelltyp"].isin([5, 7, 10, 14]).any():
        return 1
    # Non-dominant missings
    elif spells["spelltyp"].isin([11, 12, 15, 99]).any():
        return np.nan
    else:
        raise ValueError("This should not happen")
