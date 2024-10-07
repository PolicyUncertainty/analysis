import numpy as np
import pandas as pd


def merge_couples(df):
    """This function merges couples based on the 'parid' identifier.

    Partner variables are market '_p' in the merged dataset.

    """
    df = df.reset_index()
    df_partners = df.copy()

    # Assign nans to negative parids to merge nans to obs
    df_partners.loc[df_partners["parid"] < 0, "parid"] = np.nan

    merged_data = pd.merge(
        df,
        df_partners,
        how="left",
        left_on=["hid", "syear", "parid"],
        right_on=["hid", "syear", "pid"],
        suffixes=("", "_p"),
    )
    merged_data.set_index(["pid", "syear"], inplace=True)

    print(str(len(merged_data)) + " observations after merging couples.")
    return merged_data


def create_partner_state(df, filter_missing=False):
    """0: no partner, 1: working-age partner, 2: retired partner"""
    # has to be done for both state and lagged state
    # people with a partner whose choice is not observed stay in this category
    df = create_working_status(df)
    df = merge_couples(df)

    df.loc[:, "partner_state"] = np.nan
    # no partner (no parid)
    df.loc[:, "partner_state"] = np.where(df["parid"] < 0, 0, df["partner_state"])
    # working-age partner (choice 0, 1, 3)
    # Assign working partners to working state
    df.loc[:, "partner_state"] = np.where(
        df["work_status_p"] == 1, 1, df["partner_state"]
    )
    # retired partner (choice 2)
    df.loc[:, "partner_state"] = np.where(
        df["work_status_p"] == 0, 2, df["partner_state"]
    )
    if filter_missing:
        # drop nans
        df = df[df["partner_state"].notna()]
        print(
            str(len(df))
            + " observations after dropping people with a partner whose choice is not observed."
        )
    return df


def create_partner_and_lagged_state(df):
    df = create_partner_state(df)
    df["lagged_partner_state"] = df.groupby(["pid"])["partner_state"].shift()
    df = df[df["lagged_partner_state"].notna()]
    df = df[df["partner_state"].notna()]
    print(
        str(len(df))
        + " observations after dropping people with a partner whose choice is not observed."
    )
    return df


def create_working_status(df):
    df["work_status"] = np.nan
    # soep_empl_choice = df["pgemplst"]
    soep_empl_status = df["pgstib"]

    # assign emploayment choices
    df.loc[soep_empl_status != 13, "work_status"] = 1
    # df.loc[soep_empl_choice == 1, "choice"] = 1
    # df.loc[soep_empl_choice == 2, "choice"] = 3

    # assign retirement choice
    df.loc[soep_empl_status == 13, "work_status"] = 0
    # merged_df.loc[rv_ret_choice == "RTB"] = 2
    # df = df[df["choice"].notna()]
    return df
