import numpy as np


def create_float_birth_date(df, impute_missing_month=False):
    """This functions creates a float birthdate, assuming to be born on the 1st of the month.

    It uses the variables gebjahr and gebmonat from ppathl or ppath.
    If impute_missing_month is True, missing months are set to June (6). Otherwise, missing months lead to NaN birth dates.
    Missing or invalid years lead to NaN birth dates.
    """
    # Make sure, all pids have same birth year everywhere. It might be that some
    # are missing for particular years
    df["gebjahr"] = df.groupby("pid")["gebjahr"].transform("max")
    df["gebmonat"] = df.groupby("pid")["gebmonat"].transform("max")

    invalid_year = df["gebjahr"] < 0
    invalid_month = df["gebmonat"] < 0

    df["float_birth_date"] = np.nan

    if not impute_missing_month:
        valid_data = ~invalid_year & ~invalid_month
        df.loc[valid_data, "float_birth_date"] = df["gebjahr"] + df["gebmonat"] / 12
    else:
        month = df["gebmonat"].copy()
        month[invalid_month] = 6
        df.loc[~invalid_year, "float_birth_date"] = df["gebjahr"] + month / 12
    return df
