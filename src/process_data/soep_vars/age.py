import numpy as np


def calc_age_at_interview(df):
    """
    Calculate the age at interview date. Both float_interview and float_birth
    are created with missing nans for invalid data. So age will be invalid if
    one of them is invalid.
    """
    # Calculate the age at interview date
    df["float_age"] = df["float_interview"] - df["float_birth_date"]
    return df
