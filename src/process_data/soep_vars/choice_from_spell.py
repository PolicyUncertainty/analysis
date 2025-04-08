import numpy as np

from process_data.soep_vars.artkalen import prepare_artkalen_data
from process_data.soep_vars.interview_date import create_float_interview_date
from process_data.soep_vars.birth import create_float_birth_year
import matplotlib.pyplot as plt
def create_choice_variable_from_artkalen(path_dict, df):

    artkalen_data = prepare_artkalen_data(path_dict)

    df = create_float_interview_date(df)
    df = create_float_birth_year(df)
    df["choice"] = np.nan
    df["float_age"] = df["age"].astype(float)
    artkalen_data["choice"] = np.nan

    partial_select = lambda pid_group: select_spell_for_pid(pid_group, artkalen_data)
    new_df = df.groupby("pid").apply(partial_select)



    breakpoint()


def select_spell_for_pid(pid_group, artkalen_data):
    """
    Select the spells for a given pid group.
    """
    pid = pid_group.index.get_level_values("pid")[0]
    first_interview = pid_group["float_interview"].min()

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

        overlapping_spells = pid_spells[(pid_spells["float_begin"] <= interview_date )& (pid_spells["float_end"] >= interview_date)]
        choice = select_dominant_spelltyp_and_recode(overlapping_spells, pid_group.loc[id, "pgemplst"])
        pid_group.loc[id, "choice"] = choice
        if (choice == 0) & (ret_count == 0):
            start_first_ret_spell = overlapping_spells[overlapping_spells["spelltyp"] == 6].iloc[0]["float_begin"]
            pid_group.loc[id, "float_age"] = start_first_ret_spell - pid_group.iloc[0]["float_birth_year"]
            ret_count += 1
    #
    # # Assign spell type of last spell
    # start_first_spell = pid_spells.iloc[id_spells[0]]["float_begin"]
    # interviews_before = pid_group["float_interview"] < start_first_spell
    #
    # spelltyp_before = pid_spells.iloc[id_spells[0] - 1]["spelltyp"]
    # # If spelltype before is retirement. We make it absorbing by assigning it to all years
    # # and return
    # if spelltyp_before == 15:
    #     pid_group.loc[(pid, (slice(None))), "spelltyp"] = spelltyp_before
    #     return pid_group.loc[(pid, (slice(None)))]
    # else:
    #     pid_group.loc[interviews_before, "spelltyp"] = spelltyp_before
    #
    # for count_spell in range(len(id_spells) - 1):
    #     id_spell = id_spells[count_spell]
    #     spelltyp = pid_spells.iloc[id_spells[count_spell]]["spelltyp"]
    #     # Assign spell type of last spell
    #     interviews_after_spell = (
    #         pid_group["float_interview"]
    #         >= pid_spells.iloc[id_spell]["float_begin"]
    #     )
    #     if spelltyp == 15:
    #         first_ret_interview = np.where(interviews_after_spell)[0][0]
    #         float_age_start = pid_spells.iloc[id_spell]["float_begin"] - pid_group.iloc[0]["float_birth_year"]
    #         pid_group.iloc[first_ret_interview]["float_age"] = float_age_start
    #         pid_group.loc[interviews_after_spell, "spelltyp"] = spelltyp
    #         return pid_group.loc[(pid, (slice(None)))]
    #     else:
    #         interviews_before_next = (
    #             pid_group["float_interview"]
    #             < pid_spells.iloc[id_spells[count_spell + 1]]["float_begin"]
    #         )
    #         relevant_interviews_spell = interviews_after_spell & interviews_before_next
    #         pid_group.loc[relevant_interviews_spell, "spelltyp"] = spelltyp
    #
    # # Assign spell type of last spell
    # interviews_in_last_spell = (
    #     pid_group["float_interview"] >= pid_spells.iloc[id_spells[-1]]["float_begin"]
    # )
    # last_spell_typ = pid_spells.iloc[
    #     id_spells[-1]
    # ]["spelltyp"]
    # pid_group.loc[interviews_in_last_spell, "spelltyp"] = last_spell_typ
    #
    # # If spelltyp is retirement, also correct start age of first interview in last spell
    # if last_spell_typ == 15:
    #     first_ret_interview_in_last_sepll = np.where(interviews_in_last_spell)[0][0]
    #     float_age_start = pid_spells.iloc[id_spells[-1]]["float_begin"] - pid_group.iloc[0]["float_birth_year"]
    #     pid_group.iloc[first_ret_interview_in_last_sepll]["float_age"] = float_age_start

    return pid_group.loc[(pid, (slice(None)))]


def select_dominant_spelltyp_and_recode(spells, employment_status):
    """This functions implements the following rules:

        - If there is full-time employment (1) or short hours (2), we select this as full-time
        - If there is no full-time, but regular part-time (3), then we select this.
        - If there is no full- or regular part-time, but retirement (6) we select this.
        - If none of the above is there it is unemployment

    We code:
        - 0: retirement
        - 1: unemployment
        - 2: part-time
        - 3: full-time

    """
    # Treat missing data. Check if the only spelltyp is 99
    if (list(spells["spelltyp"].unique()) == [99]) | (len(spells) == 0):
        return np.nan
    elif spells["spelltyp"].isin([1, 2]).any():
       return 3
    elif spells["spelltyp"].isin([3]).any():
        return 2
    elif spells["spelltyp"].isin([6]).any():
        return 0
    elif employment_status == 1:
        return 3
    elif employment_status == 2:
        return 2
    else:
        return 1
