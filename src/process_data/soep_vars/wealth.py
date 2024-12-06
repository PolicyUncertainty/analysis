import pandas as pd
from linearmodels.panel.model import PanelOLS



def add_wealth_interpolate_and_deflate(data, path_dict, options):
    """Loads wealth data, interpolates linearly between first and last year of observation for each household, and deflates wealth using the consumer price index."""
    data = data.reset_index()
    wealth_data = load_wealth_data(path_dict["soep_c38"])
    wealth_data = trim_and_rename(wealth_data)
    wealth_data_full = interpolate_wealth(wealth_data)

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

def deflate_wealth(df, path_dict):
    """This function deflates the wealth variable using the consumer price index."""
    cpi_data = pd.read_csv(
        path_dict["intermediate_data"] + "cpi_base_2010.csv", index_col=0
    )
    df = df.merge(cpi_data, left_on="syear", right_index=True)
    df["wealth"] = df["wealth"] / df["cpi"]
    return df


def add_wealth_impute_with_panel_reg(data, path_dict, options):
    """Loads wealth data, imputes missing values using panel regression, and deflates wealth using the consumer price index."""
    data = data.reset_index()
    wealth_data = load_wealth_data(path_dict["soep_c38"])
    wealth_data = trim_and_rename(wealth_data)
    wealth_data_with_covariates, cov_list = add_hh_wealth_covariates(wealth_data, path_dict["soep_c38"])
    fitted_model = estimate_imputation_params_with_panel_ols(wealth_data_with_covariates, cov_list)
    wealth_data_full = span_full_wealth_panel(wealth_data_with_covariates, options)
    wealth_data_full_with_covariates, cov_list = add_hh_wealth_covariates(wealth_data_full, path_dict["soep_c38"])
    wealth_data_full_imputed = impute_wealth_with_panel_reg(wealth_data_full_with_covariates, fitted_model, cov_list)
    data = data.merge(wealth_data_full_imputed, on=["hid", "syear"], how="left")
    data.set_index(["pid", "syear"], inplace=True)
    print(str(len(data)) + " left after dropping people with missing wealth.")
    return data

def add_hh_wealth_covariates(df, soep_path):
    """This function gets the following household level covariates from the hl data:
    - household net incomme: hlc0005_h
    - number of children in household: hlc0043
    """
    covariate_list = ["hlc0005_h", "hlc0043"]
    hl_data = pd.read_stata(
        f"{soep_path}/hl.dta",
        columns=["hid", "syear", "hlc0005_h", "hlc0043"],
        convert_categoricals=False,
    )
    hl_data.set_index(["hid", "syear"], inplace=True)
    df.merge(hl_data, on=["hid", "syear"], how="left")
    breakpoint()
    df = df[(df["hlc0005_h"]>0) & (df["hlc0043"]>0)]
    breakpoint()
    return df, covariate_list

def estimate_imputation_params_with_panel_ols(data, cov_list):
    """Estimates the parameters of the panel regression for imputing wealth."""
    model = PanelOLS(
        dependent=data["wealth"],
        exog=data[cov_list],
        entity_effects=True,
    )
    fitted_model = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    return fitted_model

def span_full_wealth_panel(wealth_data, options):
    """Creates additional rows for each household for each year between start_year and end_year."""
    start_year = options["start_year"]
    end_year = options["end_year"]
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

def impute_wealth_with_panel_reg(wealth_data_with_covariates, fitted_model, cov_list):
    """Imputes missing values of wealth using the panel regression."""
    wealth_data_with_covariates["wealth"] = fitted_model.predict(
        wealth_data_with_covariates[cov_list]
    )
    return wealth_data_with_covariates