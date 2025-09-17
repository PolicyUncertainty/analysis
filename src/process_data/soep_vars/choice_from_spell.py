import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_data.soep_vars.artkalen import prepare_artkalen_data
from process_data.soep_vars.work_choices import create_choice_variable


def create_choice_variable_from_artkalen(
    path_dict, specs, df, load_artkalen_choice=True
):

    relevant_pids = df.index.get_level_values("pid").unique().tolist()
    artkalen_data = prepare_artkalen_data(
        path_dict, relevant_pids, specs["start_year"] - 1, specs["end_year"] + 1
    )

    if not load_artkalen_choice:
        print("Creating artkalen choice variable. This might take a while...")
        # Initialize columns
        df["choice"] = np.nan
        df["corrected_age"] = df["age"].astype(float)
        # With create artkalen choice
        partial_select = lambda pid_group: select_spell_for_pid(
            pid_group, artkalen_data
        )
        df = df.groupby("pid").apply(partial_select)
        df["art_choice"] = df["choice"].copy()
        df[["art_choice", "corrected_age"]].to_pickle(
            path_dict["struct_data"] + "art_choice.pkl"
        )
    else:
        df[["art_choice", "corrected_age"]] = pd.read_pickle(
            path_dict["struct_data"] + "art_choice.pkl"
        )

    df["lagged_art_choice"] = df.groupby("pid")["art_choice"].shift(1)

    # Create pgen choice and overwrite
    df = create_choice_variable(df, filter_missings=False)
    df["pgen_choice"] = df["choice"].copy()
    df["lagged_pgen_choice"] = df.groupby("pid")["pgen_choice"].shift(1)

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
    past_sra_retired = False
    id_first_retirement = -1
    past_ret_ids = []
    past_ret_spells = []
    float_ends_dominant_spells = []
    for interview_count in range(len(pid_group)):
        id = pid_group.index[interview_count]
        if past_sra_retired:
            pid_group.loc[id, "choice"] = 0
            continue

        interview_date = pid_group.loc[id, "float_interview"]
        if np.isnan(interview_date):
            # Take syear (second part of id) and add 0.5. This observation will be dropped later,
            # but it might be usefull for lagging or leading
            interview_date = id[1] + 0.5

        overlapping_spells = pid_spells[
            (pid_spells["float_begin"] <= interview_date)
            & (pid_spells["float_end"] >= interview_date)
        ]
        choice, float_end_dominant_spell = select_dominant_spelltyp_and_recode(
            overlapping_spells, pid_group.loc[id, "pgemplst"]
        )

        # Now lets treat retirement
        ret_choice = choice == 0

        # We first check if person already chose retirement past the retirement age
        post_sra = pid_group.loc[id, "SRA"] <= pid_group.loc[id, "age"]

        # The first time a person chooses retirement
        if ret_choice & post_sra:
            past_sra_retired = True

        pid_group.loc[id, "choice"] = choice

        # Get all covering retirement spells
        ids_overlap_ret_spells = overlapping_spells[
            overlapping_spells["spelltyp"] == 6
        ].index.values

        # Check if the retirement spell was covering a choice before, if not add
        # it to the list of new retirement spells.
        # As spells are ordered the first element of new_ret_spells will be the one
        # which started first.
        new_ret_spells = []
        for id_ret_spell in ids_overlap_ret_spells:
            if id_ret_spell not in past_ret_spells:
                new_ret_spells += [id_ret_spell]

        # If this person is not already retired, we need to manipulate the float age
        if ret_choice & (not already_retired):
            # If there is no new retirement spell we take the date
            # of the last dominant spell
            if len(new_ret_spells) == 0:
                start_ret_spell = float_ends_dominant_spells[-1]
            else:
                # If there is a new retirement spell we take the date of the first one
                start_ret_spell = overlapping_spells.loc[
                    new_ret_spells[0], "float_begin"
                ]

            pid_group.loc[id, "corrected_age"] = (
                start_ret_spell - pid_group.iloc[0]["float_birth_date"]
            )
            id_first_retirement = id
            already_retired = True

        # If retirement is chosen we add the id to the list of past retirements choices
        # We also add the id to the list of past retirement spells
        if ret_choice:
            past_ret_ids += [id]
            # Add all retirement spells to the list of past retirement spells
            for id_ret_spell in ids_overlap_ret_spells:
                if id_ret_spell not in past_ret_spells:
                    past_ret_spells += [id_ret_spell]

        # If a person is already retired, but now changes back, we need to set all past retirement
        # choices to unemployment, set already_retired to False and set the corrected_age back to
        # the even age in the first retirement observation
        if (not ret_choice) & already_retired & ~np.isnan(choice):
            for past_id in past_ret_ids:
                pid_group.loc[past_id, "choice"] = 1
            pid_group.loc[id_first_retirement, "corrected_age"] = pid_group.loc[
                id_first_retirement, "age"
            ]
            already_retired = False
            past_ret_ids = []

        # Add the end date of the dominant spell for this id to the list of end dates
        # of dominant spells
        float_ends_dominant_spells += [float_end_dominant_spell]

    return pid_group.loc[(pid, (slice(None)))]


def select_dominant_spelltyp_and_recode(spells, employment_status):
    """Select choice by spelltyp and recode it to the choice variable. Also return the end
    date of the selected spell.

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
        return np.nan, np.nan
    # Education spells
    elif spells["spelltyp"].isin([4, 8, 9, 13]).any():
        return np.nan, np.nan
    # Full-time spells
    elif spells["spelltyp"].isin([1, 2]).any():
        float_end_spells = spells[spells["spelltyp"].isin([1, 2])]["float_end"].max()
        return 3, float_end_spells
    # Retirement spells
    elif spells["spelltyp"].isin([6]).any():
        float_end_spells = spells[spells["spelltyp"].isin([6])]["float_end"].max()
        return 0, float_end_spells
    # Part-time spells
    elif spells["spelltyp"].isin([3]).any():
        float_end_spells = spells[spells["spelltyp"].isin([3])]["float_end"].max()
        return 2, float_end_spells
    # Unemployment spells
    elif spells["spelltyp"].isin([5, 7, 10, 14]).any():
        float_end_spells = spells[spells["spelltyp"].isin([5, 7, 10, 14])][
            "float_end"
        ].max()
        return 1, float_end_spells
    # Non-dominant missings
    elif spells["spelltyp"].isin([11, 12, 15, 99]).any():
        return np.nan, np.nan
    else:
        raise ValueError("This should not happen")
