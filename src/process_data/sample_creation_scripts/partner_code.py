import numpy as np
import pandas as pd


def merge_couples(df, keep_singles=False):
    """This function merges couples based on the 'parid' identifier.

    Partner variables are market '_p' in the merged dataset.

    """
    df = df.reset_index()
    df_partners = df.copy()

    if keep_singles:
        df_partners.loc[df_partners["parid"] < 0, "parid"] = np.nan
        merge_string = "left"
    else:
        merge_string = "inner"
    merged_data = pd.merge(
        df,
        df_partners,
        how=merge_string,
        left_on=["hid", "syear", "parid"],
        right_on=["hid", "syear", "pid"],
        suffixes=("", "_p"),
    )
    print(str(len(merged_data)) + " observations after merging couples.")
    return merged_data
