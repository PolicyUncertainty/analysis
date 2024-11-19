import numpy as np


def create_working_status(df):
    df["work_status"] = np.nan

    soep_empl_status = df["pgstib"]

    # assign employment choices
    df.loc[soep_empl_status != 13, "work_status"] = 1

    # assign retirement status
    df.loc[soep_empl_status == 13, "work_status"] = 0
    return df


def create_choice_variable(data):
    """This function creates the choice variable for the structural model.

    0: retirement, 1: unemployed, 2: part-time, 3: full-time

    TODO: This function assumes retirees with part-time employment as full-time retirees.

    """
    data["choice"] = np.nan
    soep_empl_choice = data["pgemplst"]
    soep_empl_status = data["pgstib"]
    # rv_ret_choice = merged_data["STATUS_2"]

    # assign employment choices
    data.loc[soep_empl_choice == 5, "choice"] = 1
    data.loc[soep_empl_choice == 2, "choice"] = 2
    data.loc[soep_empl_choice == 1, "choice"] = 3

    # assign retirement choice
    data.loc[soep_empl_status == 13, "choice"] = 0
    # merged_data.loc[rv_ret_choice == "RTB"] = 2
    data = data[data["choice"].notna()]
    return data
