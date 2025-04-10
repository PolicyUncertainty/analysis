import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

from process_data.soep_vars.artkalen import prepare_artkalen_data
from process_data.soep_vars.birth import create_float_birth_year
from process_data.soep_vars.interview_date import create_float_interview_date
from process_data.soep_vars.work_choices import create_choice_variable


def create_choice_variable_from_artkalen(path_dict, df):

    artkalen_data = prepare_artkalen_data(path_dict)

    df = create_float_interview_date(df)
    df = create_float_birth_year(df)
    df["choice"] = np.nan
    df["float_age"] = df["age"].astype(float)

    # With create artkalen choice
    # partial_select = lambda pid_group: select_spell_for_pid(pid_group, artkalen_data)
    # new_df = df.groupby("pid").apply(partial_select)
    # new_df["art_choice"] = new_df["choice"].copy()
    # pkl.dump(new_df["art_choice"], open("art_choice.pkl", "wb"))
    # new_df = create_choice_variable(new_df, filter_missings=False)

    # Load artkalen choice
    new_df = create_choice_variable(df, filter_missings=False)
    new_df["art_choice"] = pkl.load(open("art_choice.pkl", "rb"))

    nan_mask = new_df["art_choice"].isna()
    new_df.loc[nan_mask, "art_choice"] = new_df.loc[nan_mask, "choice"]

    new_df["choice"] = new_df["art_choice"].copy()

    new_df["lagged_choice"] = new_df["choice"].shift(1)

    df_fresh = new_df[
        (new_df["choice"] == 0)
        & (new_df["lagged_choice"] != 0)
        & (new_df["lagged_choice"].notna())
    ]
    # Make fine bin plot of float age
    plt.hist(df_fresh["float_age"], bins=100)

    breakpoint()


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

    ret_count = 0
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
        if (choice == 0) & (ret_count == 0):
            start_first_ret_spell = overlapping_spells[
                overlapping_spells["spelltyp"] == 6
            ].iloc[0]["float_begin"]
            pid_group.loc[id, "float_age"] = (
                start_first_ret_spell - pid_group.iloc[0]["float_birth_year"]
            )
            ret_count += 1

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
