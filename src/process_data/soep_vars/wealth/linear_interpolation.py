import numpy as np
import pandas as pd

from process_data.soep_vars.wealth.deflate_wealth import deflate_wealth


def add_wealth_interpolate_and_deflate(
    data, path_dict, specs, filter_missings=True, load_wealth=False
):
    """Loads wealth data, interpolates linearly between first and last year of
    observation for each household, and deflates wealth using the consumer price
    index."""
    # Dump wealth data or load
    file_name = path_dict["intermediate_data"] + "wealth_data.pkl"
    if load_wealth:
        wealth_data_full = pd.read_pickle(file_name)
    else:
        wealth_data = load_wealth_data(path_dict["soep_c38"])
        wealth_data_full = interpolate_and_extrapolate_wealth(
            wealth_data, path_dict, specs
        )
        # Deflate wealth
        wealth_data_full = deflate_wealth(wealth_data_full, path_dict, specs)
        # We do not allow for negative wealth values
        wealth_data_full.loc[wealth_data_full["wealth"] < 0, "wealth"] = 0
        wealth_data_full.to_pickle(file_name)

    # Now merge with existing dataset on hid and syear
    data = data.reset_index()
    data = data.merge(wealth_data_full, on=["hid", "syear"], how="left")
    data.set_index(["pid", "syear"], inplace=True)
    if filter_missings:
        data = data[(data["wealth"].notna())]
        print(str(len(data)) + " left after dropping people with missing wealth.")
    return data


def interpolate_and_extrapolate_wealth(wealth_data, path_dict, specs):
    wealth_data_full = span_full_wealth_panel(wealth_data, specs)
    # interpolate between existing points
    wealth_data_full["wealth"] = wealth_data_full.groupby("hid")["wealth"].transform(
        lambda group: group.interpolate(method="linear", limit_area="inside")
    )

    cpi_data = pd.read_csv(path_dict["open_data"] + "cpi_base_2010.csv", index_col=0)
    partial_extrapolate = lambda group: extrapolate_wealth(group, cpi_data)
    # extrapolate until the first and last observation
    wealth_data_full["wealth"] = wealth_data_full.groupby("hid").apply(
        partial_extrapolate
    )["wealth"]
    return wealth_data_full


def extrapolate_wealth(household, cpi):
    """Linearly extrapolate the wealth column at the beginning and end of each group."""
    # We use consecutive numbers for easier interpolation
    household_int = household.reset_index()
    wealth = household_int["wealth"].copy()

    # Extrapolate at the start
    if pd.isnull(wealth.iloc[0]):
        valid = wealth.dropna()
        # Get first valid index(recall it is a consecutive integer index)
        first_valid_index = valid.index[0]
        # Get the indices that are missing
        missing_start_idxs = wealth.index[wealth.index < first_valid_index]

        if len(valid) >= 2:
            # Get the two first valid values
            y = valid.iloc[:2].values
            # Slope is difference(x increment is always 1)
            slope = y[1] - y[0]
            # Fill up missing values
            wealth.loc[missing_start_idxs] = y[0] - slope * (
                first_valid_index - missing_start_idxs
            )
        elif len(valid) == 1:
            # In case of only 1 valid entry. We adjust with inflation such that the household
            # has the same real wealth throughout the years
            syears_missing = household_int.loc[missing_start_idxs, "syear"]
            valid_syear = household_int.loc[first_valid_index, "syear"]
            rescaled_cpi_of_missing = (
                cpi.loc[syears_missing, "cpi"].values / cpi.loc[valid_syear, "cpi"]
            )
            wealth.loc[missing_start_idxs] = rescaled_cpi_of_missing * valid.values
        else:
            pass

    # Extrapolate at the end
    if pd.isnull(wealth.iloc[-1]):
        # Only select valid values
        valid = wealth.dropna()
        # Get last valid index(recall it is a consecutive integer index)
        last_valid_index = valid.index[-1]
        # Get the indices that are missing
        missing_end_idx = wealth.index[wealth.index > last_valid_index]
        if len(valid) >= 2:
            # Now get the last two valid wealth values
            y = valid.iloc[-2:].values
            # Slope is difference(x increment is always 1)
            slope = y[1] - y[0]
            wealth.loc[missing_end_idx] = y[1] + slope * (
                missing_end_idx - last_valid_index
            )
        elif len(valid) == 1:
            # In case of only 1 valid entry. We adjust with inflation such that the household
            # has the same real wealth throughout the years
            syears_missing = household_int.loc[missing_end_idx, "syear"]
            valid_syear = household_int.loc[last_valid_index, "syear"]
            rescaled_cpi_of_missing = (
                cpi.loc[syears_missing, "cpi"].values / cpi.loc[valid_syear, "cpi"]
            )
            wealth.loc[missing_end_idx] = rescaled_cpi_of_missing * valid.values
        else:
            pass

    household_int["wealth"] = wealth
    household_int.set_index("syear", inplace=True)
    # household.set_index(["hid", "syear"], inplace=True)
    return household_int


def span_full_wealth_panel(wealth_data, specs):
    """Creates additional rows for each household for each year between start_year and
    end_year.

    Every household without any wealth data is dropped.

    """
    hid_values = wealth_data.index.get_level_values("hid").unique()
    min_year_wealth = wealth_data.index.get_level_values("syear").min()
    max_year_wealth = wealth_data.index.get_level_values("syear").max()
    start_year = np.minimum(min_year_wealth, specs["start_year"])
    end_year = np.maximum(max_year_wealth, specs["end_year"])

    all_index = pd.MultiIndex.from_product(
        [hid_values, range(start_year, end_year + 1)], names=["hid", "syear"]
    )
    wealth_data_full = wealth_data.reindex(all_index, fill_value=np.nan, copy=True)
    return wealth_data_full


def load_wealth_data(soep_c38_path):
    # Load SOEP core data
    wealth_data = pd.read_stata(
        f"{soep_c38_path}/hwealth.dta",
        columns=["hid", "syear", "w011ha"],
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)
    wealth_data.set_index(["hid", "syear"], inplace=True)
    wealth_data.rename(columns={"w011ha": "wealth"}, inplace=True)
    return wealth_data
