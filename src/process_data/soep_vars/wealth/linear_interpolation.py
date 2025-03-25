import pandas as pd

from process_data.soep_vars.wealth.deflate_wealth import deflate_wealth


def add_wealth_interpolate_and_deflate(data, path_dict, options):
    """Loads wealth data, interpolates linearly between first and last year of
    observation for each household, and deflates wealth using the consumer price
    index."""
    data = data.reset_index()
    wealth_data = load_wealth_data(path_dict["soep_c38"])
    wealth_data = trim_and_rename(wealth_data)
    wealth_data_full = interpolate_and_extrapolate_wealth(wealth_data, options)
    data = data.merge(wealth_data_full, on=["hid", "syear"], how="left")
    data = deflate_wealth(data, path_dict)
    data.loc[data["wealth"] < 0, "wealth"] = 0
    data.set_index(["pid", "syear"], inplace=True)
    data = data[(data["wealth"].notna())]
    print(str(len(data)) + " left after dropping people with missing wealth.")
    return data


def load_wealth_data(soep_c38_path):
    # Load SOEP core data
    wealth_data = pd.read_stata(
        f"{soep_c38_path}/hwealth.dta",
        columns=["hid", "syear", "w011ha"],
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)
    return wealth_data


def trim_and_rename(wealth_data):
    """This function trims the wealth data (no missings, negatives set to 0) and renames
    the wealth variable."""
    wealth_data = wealth_data[wealth_data["w011ha"].notna()]
    wealth_data.loc[wealth_data["w011ha"] < 0, "w011ha"] = 0
    wealth_data.rename(columns={"w011ha": "wealth"}, inplace=True)
    return wealth_data


def interpolate_wealth(wealth_data):
    # for each household, create a row for each year between min and max syear
    min_max_syear = wealth_data.groupby("hid")["syear"].agg(["min", "max"])
    all_combinations = pd.concat(
        [
            pd.DataFrame({"hid": hid, "syear": range(row["min"], row["max"] + 1)})
            for hid, row in min_max_syear.iterrows()
        ]
    )
    wealth_data_full = pd.merge(
        all_combinations, wealth_data, on=["hid", "syear"], how="left"
    )

    # Set 'hid' and 'syear' as the index
    wealth_data_full.set_index(["hid", "syear"], inplace=True)
    wealth_data_full.sort_index(inplace=True)

    # Interpolate the missing values for each household
    wealth_data_full["wealth"] = wealth_data_full.groupby("hid")["wealth"].transform(
        lambda group: group.interpolate(method="linear")
    )
    return wealth_data_full


def interpolate_and_extrapolate_wealth(wealth_data, options):
    wealth_data_full = span_full_wealth_panel(wealth_data, options)
    # interpolate between existing points
    wealth_data_full["wealth"] = wealth_data_full.groupby("hid")["wealth"].transform(
        lambda group: group.interpolate(method="linear")
    )
    # extrapolate until the first and last observation
    wealth_data_full["wealth"] = wealth_data_full.groupby("hid").apply(
        extrapolate_wealth
    )["wealth"]
    return wealth_data_full


def extrapolate_wealth(household):
    """Linearly extrapolate the wealth column at the beginning and end of each group."""
    household = household.reset_index()
    wealth = household["wealth"].copy()

    # Extrapolate at the start
    if pd.isnull(wealth.iloc[0]):
        valid = wealth.dropna()
        if len(valid) >= 2:
            x = valid.index[:2]
            y = valid.iloc[:2]
            slope = (y.iloc[1] - y.iloc[0]) / (x[1] - x[0])
            missing_start = wealth.index[wealth.index < x[0]]
            wealth.loc[missing_start] = y.iloc[0] - slope * (x[0] - missing_start)

    # Extrapolate at the end
    if pd.isnull(wealth.iloc[-1]):
        valid = wealth.dropna()
        if len(valid) >= 2:
            x = valid.index[-2:]
            y = valid.iloc[-2:]
            slope = (y.iloc[1] - y.iloc[0]) / (x[1] - x[0])
            missing_end = wealth.index[wealth.index > x[1]]
            wealth.loc[missing_end] = y.iloc[1] + slope * (missing_end - x[1])

    household["wealth"] = wealth
    household.set_index("syear", inplace=True)
    # household.set_index(["hid", "syear"], inplace=True)
    return household


def span_full_wealth_panel(wealth_data, options):
    """Creates additional rows for each household for each year between start_year and
    end_year.

    Every household without any wealth data is dropped.

    """
    start_year = options["start_year"]
    end_year = options["end_year"]
    wealth_data.set_index(["hid", "syear"], inplace=True)
    all_combinations = pd.concat(
        [
            pd.DataFrame({"hid": hid, "syear": range(start_year, end_year + 1)})
            for hid in wealth_data.index.get_level_values("hid").unique()
        ]
    )
    wealth_data_full = pd.merge(
        all_combinations, wealth_data, on=["hid", "syear"], how="left"
    )
    wealth_data_full.set_index(["hid", "syear"], inplace=True)
    wealth_data_full = wealth_data_full.groupby(level="hid").filter(
        lambda x: x["wealth"].notna().any()
    )
    return wealth_data_full
