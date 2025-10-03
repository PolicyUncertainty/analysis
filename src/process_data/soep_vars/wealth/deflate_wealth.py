import pandas as pd


def deflate_wealth(df, path_dict, specs):
    """This function deflates the wealth variable using the consumer price index.

    We retrieved the cpi with base 10 at Statistisches Bundesamt (Destatis), 2025 | Stand: 25.03.2025
    on https://www-genesis.destatis.de/datenbank/online/statistic/61111/table/61111-0001/table-toolbar
    """
    cpi_data = pd.read_csv(path_dict["open_data"] + "cpi_base_2010.csv", index_col=0)
    # We need to set the index to the year
    cpi_data /= cpi_data.loc[specs["reference_year"]]
    df = df.merge(cpi_data, left_on="syear", right_index=True)
    df["wealth"] = df["wealth"] / df["cpi"]
    return df
