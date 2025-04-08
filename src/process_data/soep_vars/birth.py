import numpy as np


def create_float_birth_year(df, drop_missing_month=True):
    """This functions creates a float birthdate, assuming to be born on the 1st of the month.

    It uses the variables gebjahr and gebmonat from ppathl or ppath.
    It allows to specify to drop missing month or otherwise use June/mid year
    instead.

    """

    invalid_year = df["gebjahr"] < 0
    invalid_month = df["gebmonat"] < 0

    df["float_birth_year"] = np.nan

    if drop_missing_month:
        valid_data = ~invalid_year & ~invalid_month
        df.loc[valid_data, "float_birth_year"] = df["gebjahr"] + df["gebmonat"] / 12
    else:
        month = df["gebmonat"]
        month[invalid_month] = 6
        df.loc[~invalid_year, "float_birth_year"] = df["gebjahr"] + month / 12

    return df
