import numpy as np


def create_float_interview_date(df):
    """
    Create a float variable for the interview date in the format YYYY.MM.
    """
    months_array = df["pmonin"].values
    months_array[np.isnan(months_array)] = -1
    ivalid_mask = months_array == 0
    months_array[ivalid_mask] = -1
    months_array = months_array.astype(int)

    days_up_to_month = create_n_days_for_month(months_array)
    total_days = days_up_to_month + df["ptagin"]

    total_invalid_mask = (df["ptagin"].values == 0) | ivalid_mask
    total_days[total_invalid_mask] = np.nan
    df["float_interview"] = df.index.get_level_values("syear").values + total_days / 365
    return df


def create_n_days_for_month(current_months):
    """
    Create a variable for sum of number of days excluding current month.
    Code missing months as -1 days

    Args:

        -- current_months (int): Array of months with negative values for missing
    """
    # Create a list of the number of days excluding the current month
    n_days = np.array(
        [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=int
    )
    days_up_to_month = n_days[current_months - 1]
    days_up_to_month[current_months < 0] = -1

    return days_up_to_month
