import pandas as pd


def deflate_wealth(df, path_dict):
    """This function deflates the wealth variable using the consumer price index."""
    cpi_data = pd.read_csv(path_dict["open_data"] + "cpi_base_2010.csv", index_col=0)
    df = df.merge(cpi_data, left_on="syear", right_index=True)
    df["wealth"] = df["wealth"] / df["cpi"]
    return df
