import numpy as np

from process_data.soep_vars.birth import create_float_birth_date
from process_data.soep_vars.interview_date import create_float_interview_date


def calc_age_at_interview(df):
    """
    Calculate the age at interview date. Both float_interview and float_birth
    are created with missing nans for invalid data. So age will be invalid if
    one of them is invalid.
    """
    # Create birth and interview date
    df = create_float_interview_date(df)
    df = create_float_birth_date(df)

    # Calculate the age at interview date
    df["float_age"] = df["float_interview"] - df["float_birth_date"]
    df["age"] = np.floor(df["float_age"])
    return df
