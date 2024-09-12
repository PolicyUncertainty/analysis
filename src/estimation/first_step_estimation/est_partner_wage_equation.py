# Description: This file estimates the parameters of the HOURLY wage equation using the SOEP panel data.
# We estimate the following equation for each education level:
# ln_partner_wage = beta_0 + beta_1 * ln(age) individual_FE + time_FE + epsilon

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS
from model_code.derive_specs import read_and_derive_specs


def estimate_partner_wage_parameters(paths_dict, est_men):
    """Estimate the wage parameters partners by education group in the sample.
    Est_men is a boolean that determines whether the estimation is done
    for men or for women.
    """
    wage_data = prepare_estimation_data(paths_dict, est_men=est_men)

    # Initialize empty container for coefficients
    coefficients = [0] * len(wage_data["education"].unique())
    for education in wage_data["education"].unique():
        wage_data_edu = wage_data[wage_data["education"] == education]
        # estimate parametric regression, save parameters
        model = PanelOLS(
            dependent=wage_data_edu["ln_partner_hrl_wage"],
            exog=wage_data_edu[["constant", "ln_age", "year"]],
            entity_effects=True,
        )
        fitted_model = model.fit(
            cov_type="clustered", cluster_entity=True, cluster_time=True
        )
        coefficients[education] = fitted_model.params[0:2]
        income_shock_std = np.sqrt(fitted_model.resids.var())
        coefficients[education].loc["income_shock_std"] = income_shock_std
        coefficients[education].name = education

    wage_parameters = pd.DataFrame(coefficients)

    append = "men" if est_men else "women"
    out_file_path = paths_dict["est_results"] + f"partner_wage_eq_params_{append}.csv"
    wage_parameters.to_csv(out_file_path)
    return coefficients


def prepare_estimation_data(paths_dict, est_men):
    """Prepare the data for the wage estimation."""
    # load and modify data
    wage_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_wage_estimation_sample.pkl"
    )
    if est_men:
        sex_var = 0
    else:
        sex_var = 1

    wage_data = wage_data[wage_data["sex"] == sex_var]

    # own data
    wage_data["ln_age"] = np.log(wage_data["age"])

    # partner data
    wage_data["ln_partner_hrl_wage"] = np.log(wage_data["hourly_wage_p"])

    # prepare format
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    wage_data["constant"] = np.ones(len(wage_data))

    return wage_data


def calculate_partner_hours(path_dict):
    """Calculates average hours worked by working partners (i.e. conditional on working hours > 0)
    Produces partner_hours array of shape (n_sexes, n_education_types, n_working_periods)"""
    specs = read_and_derive_specs(path_dict["specs"])
    start_age = specs["start_age"]
    end_age = specs["max_ret_age"]
    # load data, filter, create age bins 
    df = pd.read_pickle(path_dict["intermediate_data"] + "partner_wage_estimation_sample.pkl")
    df = df[df["age"] >= start_age]
    df = df[df["age"] <= end_age]
    df["age_bin"] = np.floor(df["age"] / 10) * 10
    df.loc[df["age"] > 60, "age_bin"] = 60
    df["period"] = df["age"] - start_age

    # calculate average hours worked by partner by age, sex and education
    cov_list = ["sex", "education", "age_bin"]
    partner_hours = df.groupby(cov_list)["working_hours_p"].mean()

    # save to csv
    out_file_path = path_dict["est_results"] + f"partner_hours.csv"
    partner_hours.to_csv(out_file_path)
    return partner_hours    

