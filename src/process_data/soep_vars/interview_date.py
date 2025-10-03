import numpy as np
import pandas as pd


def create_float_interview_date(df, impute_missing_interview=False):
    """Create float interview date (YYYY.fraction) from syear, pmonin, ptagin.
    If impute_missing_interview is True, missing months are imputed using the mode
    of valid interviews per year, and missing days are set to 15.
    Otherwise, missing months or days lead to NaN interview dates.
    """

    # Work with copies to avoid modifying original data
    month = df["pmonin"].copy()
    day = df["ptagin"].copy()

    if impute_missing_interview:
        month, day = _impute_missing_interview_dates(df, month, day)

    # Check for remaining invalid data
    invalid_month = month.isna() | (month < 1) | (month > 12)
    invalid_day = day.isna() | (day < 1) | (day > 31)
    any_invalid = invalid_month | invalid_day

    # Calculate days from year start (only for valid data)
    total_days = pd.Series(np.nan, index=df.index)
    valid_mask = ~any_invalid

    if valid_mask.any():
        days_up_to_month = _cumulative_days_before_month(month[valid_mask])
        total_days[valid_mask] = days_up_to_month + day[valid_mask]

    # Create float date - handle syear in index or columns
    if "syear" in df.columns:
        year_values = df["syear"]
    elif "syear" in df.index.names:
        year_values = df.index.get_level_values("syear")
    else:
        raise ValueError("syear not found in columns or index")

    df["float_interview"] = year_values + total_days / 365
    return df


def _cumulative_days_before_month(months):
    """Get cumulative days before each month (Jan=0, Feb=31, Mar=59, etc.)"""
    days_before = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    # Convert to numpy array for vectorized indexing, then back to Series
    month_indices = months.astype(int) - 1  # Convert to 0-based indexing
    return pd.Series(days_before[month_indices], index=months.index)


def _impute_missing_interview_dates(df, month, day):
    """Impute missing interview dates using mode of valid interviews per year"""

    # Create valid month mask
    valid_month_mask = df["pmonin"].notna() & (df["pmonin"] >= 1) & (df["pmonin"] <= 12)

    if valid_month_mask.any():
        # Get syear values regardless of whether it's in index or columns
        if "syear" in df.columns:
            syear_values = df.loc[valid_month_mask, "syear"]
        elif "syear" in df.index.names:
            syear_values = df.index.get_level_values("syear")[valid_month_mask]
        else:
            raise ValueError("syear not found in columns or index")

        # Create a simple DataFrame for grouping (strip index to avoid conflicts)
        valid_months_df = pd.DataFrame({
            'pmonin': df.loc[valid_month_mask, "pmonin"].values,
            'syear': syear_values.values
        })

        # Vectorized mode calculation per year
        mode_month_by_year = (valid_months_df.groupby("syear")["pmonin"]
                             .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]))

        # For years with no valid data, use overall mode or fallback to 6
        if len(mode_month_by_year) > 0:
            overall_mode = valid_months_df["pmonin"].mode()
            fallback_month = overall_mode.iloc[0] if not overall_mode.empty else 6
        else:
            fallback_month = 6
    else:
        # No valid months at all, create empty series and use fallback
        mode_month_by_year = pd.Series(dtype=float)
        fallback_month = 6

    # Get year values for mapping (handle both cases)
    if "syear" in df.columns:
        year_values = df["syear"]
    elif "syear" in df.index.names:
        year_values = df.index.get_level_values("syear")
    else:
        raise ValueError("syear not found in columns or index")

    imputed_months = year_values.map(mode_month_by_year).fillna(fallback_month)

    # Impute missing months
    missing_month = month.isna() | (month < 1) | (month > 12)
    month = month.where(~missing_month, imputed_months)

    # Impute missing days with 15th (simple vectorized operation)
    missing_day = day.isna() | (day <= 0)
    day = day.where(~missing_day, 15)

    return month, day