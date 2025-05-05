import numpy as np
import pandas as pd

from process_data.soep_vars.wealth.deflate_wealth import deflate_wealth
from process_data.aux_and_plots.filter_data import filter_below_age
from process_data.soep_vars.age import calc_age_at_interview

def add_wealth_interpolate_and_deflate(
    data, path_dict, specs, filter_missings=True, load_wealth=False, use_processed_pl=True
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
        wealth_data_full = span_full_wealth_panel(wealth_data, specs)

        # Merge wealth data with pid/syear information
        wealth_data_full = add_personal_data(path_dict, specs, wealth_data_full, use_processed_pl)

        # Interpolate wealth for each household (consistent hh size)
        wealth_data_full = interpolate_and_extrapolate_wealth(wealth_data_full)
        breakpoint()
        # Deflate wealth
        wealth_data_full = deflate_wealth(wealth_data_full, path_dict)
        # We do not allow for negative wealth values
        wealth_data_full.loc[wealth_data_full["wealth"] < 0, "wealth"] = 0
        wealth_data_full.to_pickle(file_name)
    
    breakpoint()
    # Now merge with existing dataset on hid and syear
    data = data.reset_index()
    data = data.merge(wealth_data_full, on=["hid", "syear"], how="left")
    data.set_index(["pid", "syear"], inplace=True)
    if filter_missings:
        data = data[(data["wealth"].notna())]
        print(str(len(data)) + " left after dropping people with missing wealth.")
    return data


def interpolate_and_extrapolate_wealth(wealth_data_full):

    # hid 167, 302, 930, 981, 1031, 2046, 5240, 9474, 3490091, 3503398 show some edge cases and how they are handled... 
    # breakpoint()
    # wealth_data_full.loc[167] 
    # wealth_data_full.loc[302]
    # wealth_data_full.loc[930]
    # wealth_data_full.loc[981]
    # wealth_data_full.loc[1031]
    # wealth_data_full.loc[2046]
    # wealth_data_full.loc[5240]
    # wealth_data_full.loc[9474]
    # wealth_data_full.loc[3490091]
    # wealth_data_full.loc[3503398]

    # interpolate between existing points with consistent household size (only people specs["start_age"] or older)
    wealth_data_full["hh_size_adjusted"] = wealth_data_full.groupby(["hid", "syear"])["pid"].count() 
    # only keep households with ppathl data for at least one person in a given year 
    wealth_data_full = wealth_data_full[wealth_data_full["hh_size_adjusted"] > 0]
    # in households with 2+ people above start age drop people with no partner in the household
    # this removes e.g. - child thats older than start age and living with parents
    #                   - a person that is not a partner and living with a couple e.g. a retired parent 
    #                   - shared apartment (e.g. two/three/four friends above start age)
    # around 700 household are effected with 7500 observations being dropped (out of 360k)
    wealth_data_full = wealth_data_full[~((wealth_data_full["hh_size_adjusted"] >= 2) & (wealth_data_full["is_par"] == 0))]
    # recalculate household size without the dropped people
    wealth_data_full["hh_size_adjusted"] = wealth_data_full.groupby(["hid", "syear"])["pid"].count()
    # drop households with more than 2 people above start age left. This happens e.g. if two or more couples live together or a married person moves in with their parents doing care work etc. 
    # this happens in around 80 households with 700 observations being dropped
    wealth_data_full = wealth_data_full[wealth_data_full["hh_size_adjusted"] <= 2]

    # Interpolate wealth between observations
    wealth_data_full["wealth"] = wealth_data_full.groupby(["hid", "hh_size_adjusted", "pid"])["wealth"].transform(lambda group: group.interpolate(method="linear", limit_area="inside"))

    # Keep hh if the hh (by size and pid) has at least one non-NaN wealth value for a syear, mask == NaN if is_par == NaN => drop rows, since they are not in the panel
    mask = wealth_data_full.groupby(["hid", "hh_size_adjusted", "pid"])["wealth"].transform(lambda x: x.notna().any())
    mask &= wealth_data_full["is_par"].notna()
    wealth_data_full = wealth_data_full[mask]
    # find mean hh age for each household (rounded to nearest integer)
    wealth_data_full["mean_household_age"] = wealth_data_full.groupby(["hid", "syear"])["float_age"].transform(lambda x: x.mean().round(0))

    # create a dataframe with percentiles (1% to 100%) of wealth, grouped by each unique combination of: syear, mean_household_age, hh_size_adjusted
    wealth_data_unique = wealth_data_full.reset_index().drop_duplicates(subset=['hid', 'syear'])
    wealth_percentiles_df = (
        wealth_data_unique
        .groupby(['syear', 'mean_household_age', 'hh_size_adjusted'])
        .apply(compute_percentiles)
        .reset_index()
        .set_index(['syear', 'mean_household_age', 'hh_size_adjusted'])
    )

    # extrapolate wealth for each household at the start and end of the panel if there are at least 2 valid observations (by pid too in case household goes from A to AB to B)
    partial_extrapolate = lambda group: extrapolate_wealth(group, wealth_percentiles_df, wealth_data_full)
    extrapolated = wealth_data_full.groupby(["hid", "hh_size_adjusted", "pid"]).apply(partial_extrapolate)
    # drop the pid column and keep only the wealth column
    extrapolated = extrapolated.drop(columns=["hid", "hh_size_adjusted", "pid"])
    extrapolated = extrapolated.reset_index()
    wealth_data_full = wealth_data_full.reset_index().drop(columns="wealth").merge(
        extrapolated[["hid", "syear", "wealth"]], on=["hid", "syear"], how="left"
    ).set_index(["hid", "syear", "pid"])

    # drop duplicates caused by doing the extrapolation twice in hh with 2 people
    wealth_data_full = wealth_data_full[~wealth_data_full.index.duplicated(keep="first")]

    return wealth_data_full


def extrapolate_wealth(household, wealth_percentiles_df, wealth_data_full):
    """Linearly extrapolate the wealth column at the beginning and end of each group."""
    # We use consecutive numbers for easier interpolation
    household_int = household.reset_index()
    wealth = household_int["wealth"].copy()
    syear = household_int["syear"].copy()

    # Extrapolate at the start
    if pd.isnull(wealth.iloc[0]):
        valid = wealth.dropna()
        # Get first valid index (recall it is a consecutive integer index)
        first_valid_index = valid.index[0]
        # Get the indices that are missing
        missing_start_idxs = wealth.index[wealth.index < first_valid_index]

        if len(valid) >= 2:
            # Get the two first valid values
            y = valid.iloc[:2].values
            x = syear.iloc[valid.iloc[:2].index].values
            # Slope is difference in y divided by difference x
            slope = y[1] - y[0] / (x[1] - x[0])
            # Fill up missing values
            wealth.loc[missing_start_idxs] = y[0] - slope * (
                first_valid_index - missing_start_idxs
            )
        elif len(valid) == 1:
            pass

    # Extrapolate at the end
    if pd.isnull(wealth.iloc[-1]):
        # Only select valid values
        valid = wealth.dropna()
        # Get last valid index (recall it is a consecutive integer index)
        last_valid_index = valid.index[-1]
        # Get the indices that are missing
        missing_end_idx = wealth.index[wealth.index > last_valid_index]
        if len(valid) >= 2:
            # Now get the last two valid wealth values
            y = valid.iloc[-2:].values
            x = syear.iloc[valid.iloc[-2:].index].values
            # Slope is difference in y divided by difference x
            slope = y[1] - y[0] / (x[1] - x[0])
            # Fill up missing values    
            wealth.loc[missing_end_idx] = y[1] + slope * (
                missing_end_idx - last_valid_index
            )

    household_int["wealth"] = wealth
    household_int.set_index("syear", inplace=True)
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

def compute_percentiles(group):
    return pd.Series(
        np.percentile(group['wealth'], np.arange(1, 101)),  # Percentiles 1 to 100
        index=[f'w_pctl_{i}' for i in range(1, 101)]
    )
    
def load_wealth_data(soep_c38_path):
    # Load SOEP core data
    wealth_data = pd.read_stata(
        f"{soep_c38_path}/hwealth.dta",
        columns=["hid", "syear", "w011ha"], # "w011hb", "w011hc", "w011hd", "w020h0" 
        convert_categoricals=False,
    )
    wealth_data["hid"] = wealth_data["hid"].astype(int)
    wealth_data.set_index(["hid", "syear"], inplace=True)
    wealth_data.rename(columns={"w011ha": "wealth"}, inplace=True)
    return wealth_data

def add_personal_data(path_dict, specs, wealth_data_full, use_processed_pl=True):
    """Load ppathl data from SOEP."""
    soep_c38_path = path_dict["soep_c38"]
    # Start with ppathl. Everyone is in there even if not individually surveyed and just member
    # of surveyed household
    ppathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["pid", "hid", "syear", "sex", "parid", "gebjahr", "gebmonat"], # "rv_id", ... 
        convert_categoricals=False,
    )
    ppathl_data.dropna(inplace=True) # drop if most basic data is missing
    ppathl_data["hid"] = ppathl_data["hid"].astype(int)

    pl_intermediate_file = path_dict["intermediate_data"] + "pl_structural_w.pkl"
    if use_processed_pl:
        pl_data = pd.read_pickle(pl_intermediate_file)
    else:
        # Add pl data
        pl_data_reader = pd.read_stata(
            f"{soep_c38_path}/pl.dta",
            columns=["pid", "hid", "syear", "pmonin", "ptagin"], # "plb0304_h", "iyear", ... 
            chunksize=100000,
            convert_categoricals=False,
        )
        pl_data = pd.DataFrame()
        for itm in pl_data_reader:
            pl_data = pd.concat([pl_data, itm])
        
        pl_data["hid"] = pl_data["hid"].astype(int)
        pl_data.to_pickle(pl_intermediate_file)

    merged_data = pd.merge(ppathl_data, pl_data, on=["pid", "syear", "hid"], how="left")

    # set index to pid and syear, create age and filter by age, drop pids 
    merged_data.set_index(["pid", "syear"], inplace=True)
    merged_data = calc_age_at_interview(merged_data)
    merged_data = filter_below_age(merged_data, specs["start_age"])
    merged_data["is_par"] = np.where(
        merged_data["parid"] == -2, 0, 1
    )
    merged_data.reset_index(inplace=True)

    # add number of children in household to merged_data
    pequiv_data = pd.read_stata(
        # d11107: number of children in household
        f"{soep_c38_path}/pequiv.dta",
        columns=["pid", "syear", "d11107"],
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="left")
    merged_data.rename(columns={"d11107": "children"}, inplace=True)
    merged_data.reset_index(inplace=True)
    merged_data.set_index(["hid", "syear"], inplace=True)

    # Merge wealth data with hid/syear information and return 
    return wealth_data_full.merge(merged_data, right_index=True, left_index=True, how="left")