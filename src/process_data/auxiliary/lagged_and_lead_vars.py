import numpy as np
import pandas as pd


def _print_filter(before, after, msg):
    pct = (after - before) / before * 100 if before > 0 else 0
    print(f"{after} {msg} ({pct:+.2f}%)")


def span_dataframe(df, start_year, end_year):
    """This function spans the DataFrame over the whole observation period from
    start_year to end_year, to create lagged and lead variables."""
    # Create full index with all possible combinations of pid and syear. Otherwise if
    # we just shift the data, people having missing years in their observations get
    # assigned variables from multi years back.
    before = len(df)
    pid_indexes = df.index.get_level_values(0).unique()

    full_index = pd.MultiIndex.from_product(
        [pid_indexes, range(start_year, end_year + 1)],
        names=["pid", "syear"],
    )
    full_df = df.reindex(full_index)

    if "hid" in full_df.columns.values:
        full_df["hid"] = full_df.groupby(["pid"])["hid"].transform("last")

    _print_filter(before, len(full_df), "observations after spanning dataframe")
    return full_df
