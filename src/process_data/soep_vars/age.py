import numpy as np

from process_data.soep_vars.birth import create_float_birth_date
from process_data.soep_vars.interview_date import create_float_interview_date


def calc_age_at_interview(df, impute_missing_month=True, impute_missing_interview=True, drop_invalid=False):
    """
    Calculate the age at interview date. Both float_interview and float_birth
    are created with missing nans for invalid data. So age will be invalid if
    one of them is invalid.

    Vars needed: gebjahr, gebmonat, syear, pmonin, ptagin
    """
    before = len(df)

    # Create birth and interview date
    df = create_float_interview_date(df, impute_missing_interview=impute_missing_interview)
    df = create_float_birth_date(df, impute_missing_month=impute_missing_month)

    # Calculate the age at interview date
    df["float_age"] = df["float_interview"] - df["float_birth_date"]
    df["age"] = np.floor(df["float_age"])

    if drop_invalid:
        df = df[df["age"].notna() & (df["age"] >= 0)].copy()
        _print_filter(before, len(df), "left after dropping invalid age data (missing birth month or birth year)")
    return df

def _print_filter(before, after, msg):
    pct = (after - before) / before * 100 if before > 0 else 0
    print(f"{after} {msg} ({pct:+.2f}%)")