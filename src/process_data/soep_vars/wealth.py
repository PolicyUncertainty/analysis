import pandas as pd
from linearmodels.panel.model import PanelOLS



def add_wealth_interpolate_and_deflate(data, path_dict, options):
    """Loads wealth data, interpolates linearly between first and last year of observation for each household, and deflates wealth using the consumer price index."""
    data = data.reset_index()
    wealth_data = load_wealth_data(path_dict["soep_c38"])
    wealth_data = trim_and_rename(wealth_data)
    wealth_data_full = interpolate_and_extrapolate_wealth(wealth_data, options)
    data = data.merge(wealth_data_full, on=["hid", "syear"], how="left")
    data = deflate_wealth(data, path_dict)
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
    """This function trims the wealth data (no missings, negatives set to 0) and renames the wealth variable."""
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
    print(len(wealth_data_full))
    breakpoint()
    wealth_data_full["wealth"] = wealth_data_full.groupby("hid").apply(extrapolate_wealth)
    print(len(wealth_data_full))
    return wealth_data_full

def extrapolate_wealth(household):
    """
    Linearly extrapolate the wealth column at the beginning and end of each group.
    """
    household = household.reset_index()
    wealth = household['wealth'].copy()

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
    
    household.set_index(["hid", "syear"], inplace=True)
    return wealth

def span_full_wealth_panel(wealth_data, options):
    """Creates additional rows for each household for each year between start_year and end_year."""
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
    return wealth_data_full

def deflate_wealth(df, path_dict):
    """This function deflates the wealth variable using the consumer price index."""
    cpi_data = pd.read_csv(
        path_dict["intermediate_data"] + "cpi_base_2010.csv", index_col=0
    )
    df = df.merge(cpi_data, left_on="syear", right_index=True)
    df["wealth"] = df["wealth"] / df["cpi"]
    return df


# deprecated functions since we use simple linear interpolation and extrapolation

def add_wealth_impute_with_panel_reg(data, path_dict, options):
    """Loads wealth data, imputes missing values using panel regression.
    Note: we do not de-/ inflate the wealth variable but estimate a linear time trend and predict using a reference year.
    """
    data = data.reset_index()
    wealth_data = load_wealth_data(path_dict["soep_c38"])
    wealth_data = trim_and_rename(wealth_data)
    wealth_data_with_covariates, cov_list = add_hh_wealth_covariates(wealth_data, path_dict["soep_c38"])
    fitted_model = estimate_imputation_params_with_panel_ols(wealth_data_with_covariates, cov_list, options)
    wealth_data_full = span_full_wealth_panel(wealth_data_with_covariates, options)
    wealth_data_full_with_covariates, cov_list = add_hh_wealth_covariates(wealth_data_full, path_dict["soep_c38"])
    wealth_data_full_imputed = impute_wealth_with_panel_reg(wealth_data_full_with_covariates, fitted_model, cov_list, options)

    data = data.merge(wealth_data_full_imputed, on=["hid", "syear"], how="left")
    data.set_index(["pid", "syear"], inplace=True)
    print(str(len(data)) + " left after dropping people with missing wealth.")
    return data

def add_hh_wealth_covariates(df, soep_path):
    """This function gets the following household level covariates from the hl data:
    - household net income: hlc0005_h
    """
    covariate_list = ["hlc0005_h"]
    columns = ["hid", "syear"] + covariate_list
    hl_data = pd.read_stata(
        f"{soep_path}/hl.dta",
        columns=columns,
        convert_categoricals=False,
    )
    hl_data.set_index(["hid", "syear"], inplace=True)
    df = df.drop(columns=covariate_list, errors="ignore")
    df = df.merge(hl_data, on=["hid", "syear"], how="left")
    df = df[df["hlc0005_h"]>0]
    return df, covariate_list

def estimate_imputation_params_with_panel_ols(data, cov_list, options):
    """Estimates the parameters of the panel regression for imputing wealth with a linear time trend."""
    data.set_index(["hid", "syear"], inplace=True)
    data["year"] = data.index.get_level_values("syear") - options["start_year"]
    cov_list = cov_list + ["year"]
    model = PanelOLS(
        dependent=data["wealth"],
        exog=data[cov_list],
        entity_effects=True,
    )
    fitted_model = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    return fitted_model



def impute_wealth_with_panel_reg(wealth_data_with_covariates, fitted_model, cov_list, options):
    """Imputes missing values of wealth using the panel regression, de-/ or inflating the wealth variable by setting the year to the reference year."""
    # debugging
    entity_effects = fitted_model.estimated_effects.unstack(level=1).mean(axis=1)

    # set year to reference year
    wealth_data_with_covariates["year"] = options["reference_year"] - options["start_year"]
    exog = cov_list + ["year"]
    # predict wealth
    wealth_data_with_covariates["wealth_predicted"] = fitted_model.predict(
        wealth_data_with_covariates[exog]
    )
    wealth_data_with_covariates["wealth"] = wealth_data_with_covariates["wealth"].combine_first(
        wealth_data_with_covariates["wealth_predicted"]
    )
    return wealth_data_with_covariates