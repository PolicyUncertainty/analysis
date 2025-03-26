import numpy as np
import pandas as pd


def span_dataframe(df, start_year, end_year):
    """This function spans the DataFrame over the whole observation period from
    start_year to end_year, to create lagged and lead variables."""
    # Create full index with all possible combinations of pid and syear. Otherwise if
    # we just shift the data, people having missing years in their observations get
    # assigned variables from multi years back.
    pid_indexes = df.index.get_level_values(0).unique()

    full_index = pd.MultiIndex.from_product(
        [pid_indexes, range(start_year, end_year + 1)],
        names=["pid", "syear"],
    )
    full_container = pd.DataFrame(
        index=full_index, data=np.nan, dtype=float, columns=df.columns
    )
    full_container.update(df)

    if "hid" in full_container.columns.values:
        full_container["hid"] = full_container.groupby(["pid"])["hid"].transform("last")
    return full_container


def create_lagged_and_lead_variables(
    data, specs, lead_job_sep=False, filter_missings=True
):
    """This function creates the lagged choice variable and drops missing lagged
    choices."""

    df_full = span_dataframe(data, specs["start_year"] - 1, specs["end_year"] + 1)

    df_full["lagged_choice"] = df_full.groupby(["pid"])["choice"].shift()

    if lead_job_sep:
        df_full["job_sep_this_year"] = df_full.groupby(["pid"])["job_sep"].shift(-1)

    if filter_missings:
        if lead_job_sep:
            df_full = df_full[df_full["job_sep_this_year"].notna()]

        df_full = df_full[df_full["lagged_choice"].notna()]
        # We now have observations with a valid lagged or lead variable but not with
        # actual valid state variables. Delete those by looking at the choice variable.
        df_full = df_full[df_full["choice"].notna()]

        print(str(len(df_full)) + " left after filtering missing lagged choices.")
    return df_full
