import numpy as np

from process_data.soep_vars.artkalen import prepare_artkalen_data
from process_data.soep_vars.interview_date import create_float_interview_date


def create_choice_variable_from_artkalen(path_dict, df):

    artkalen_data = prepare_artkalen_data(path_dict)

    df = create_float_interview_date(df)
    df["spelltyp"] = np.nan

    partial_select = lambda pid_group: select_spell_for_pid(pid_group, artkalen_data)

    new_df = df.groupby("pid").apply(partial_select)

    breakpoint()


def select_spell_for_pid(pid_group, artkalen_data):
    """
    Select the spells for a given pid group.
    """
    pid = pid_group.index.get_level_values("pid")[0]
    # Select pid spells of arkalen
    pid_spells = artkalen_data[artkalen_data["pid"] == pid].sort_values("float_begin")
    # Select first interview and relevant spells
    first_interview = pid_group["float_interview"].min()
    id_spells = np.where(pid_spells["float_begin"] >= first_interview)[0]

    if len(id_spells) == 0:
        return pid_group.loc[(pid, (slice(None)))]

    # Assign spell type of last spell
    start_first_spell = pid_spells.iloc[id_spells[0]]["float_begin"]
    interviews_before = pid_group["float_interview"] < start_first_spell

    spelltyp_before = pid_spells.iloc[id_spells[0] - 1]["spelltyp"]
    # If spelltype before is retirement. We make it absorbing by assigning it to all years
    # and return
    if spelltyp_before == 15:
        pid_group.loc[(pid, (slice(None))), "spelltyp"] = spelltyp_before
        return pid_group.loc[(pid, (slice(None)))]
    else:
        pid_group.loc[interviews_before, "spelltyp"] = spelltyp_before

    for count_spell in range(len(id_spells) - 1):
        spelltyp = pid_spells.iloc[id_spells[count_spell]]["spelltyp"]
        # Assign spell type of last spell
        interviews_after_spell = (
            pid_group["float_interview"]
            >= pid_spells.iloc[id_spells[count_spell]]["float_begin"]
        )
        if spelltyp == 15:
            first_ret_interview = np.where(interviews_after_spell)[0][0]

        interviews_before_next = (
            pid_group["float_interview"]
            < pid_spells.iloc[id_spells[count_spell + 1]]["float_begin"]
        )
        relevant_interviews_spell = interviews_after_spell & interviews_before_next
        pid_group.loc[relevant_interviews_spell, "spelltyp"] = spelltyp

    # Assign spell type of last spell
    interviews_in_last_spell = (
        pid_group["float_interview"] >= pid_spells.iloc[id_spells[-1]]["float_begin"]
    )
    pid_group.loc[interviews_in_last_spell, "spelltyp"] = pid_spells.iloc[
        id_spells[-1]
    ]["spelltyp"]

    return pid_group.loc[(pid, (slice(None)))]
