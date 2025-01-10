# Description: This file estimates the parameters of the HOURLY wage equation using the SOEP panel data.
# We estimate the following equation for each education level:
# ln_partner_wage = beta_0 + beta_1 * ln(age) individual_FE + time_FE + epsilon
import numpy as np
import pandas as pd
import statsmodels.api as sm
from specs.derive_specs import read_and_derive_specs


def estimate_partner_wage_parameters(paths_dict, specs, est_men):
    """Estimate the wage parameters partners by education group in the sample.

    Est_men is a boolean that determines whether the estimation is done for men or for
    women.

    """
    wage_data = prepare_estimation_data(paths_dict, specs, est_men=est_men)

    edu_labels = specs["education_labels"]
    model_params = ["constant", "period", "period_sq"]
    # Initialize empty container for coefficients
    wage_parameters = pd.DataFrame(
        index=pd.Index(data=edu_labels, name="education"),
        columns=model_params,
    )

    for edu_val, edu_label in enumerate(edu_labels):
        # Filter df
        wage_data_edu = wage_data[wage_data["education"] == edu_val]
        wage_data_edu = sm.add_constant(wage_data_edu)
        # make ols regression
        model = sm.OLS(
            endog=wage_data_edu["wage_p"],
            exog=sm.add_constant(wage_data_edu[["constant", "period", "period_sq"]]),
            missing="drop",
        )
        fitted_model = model.fit()

        # Assign estimated parameters (column list corresponds to model params, so only these are assigned)
        wage_parameters.loc[edu_label] = fitted_model.params

        # # Get estimate for income shock std
        # income_shock_std = np.sqrt(fitted_model.resids.var())
        # wage_parameters.loc[education, "income_shock_std"] = income_shock_std

        # # Plot fitted values
        # wage_data_edu["wage_pred"] = fitted_model.predict()
        # wage_data_edu.groupby("age")["wage_pred"].mean().plot()
        # wage_data_edu.groupby("age")["wage_p"].mean().plot()
        # import matplotlib.pyplot as plt
        #
        # plt.show()

    append = "men" if est_men else "women"
    out_file_path = paths_dict["est_results"] + f"partner_wage_eq_params_{append}.csv"
    wage_parameters.to_csv(out_file_path)
    return wage_parameters


def prepare_estimation_data(paths_dict, specs, est_men):
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

    # Add period
    wage_data["period"] = wage_data["age"] - specs["start_age"]
    wage_data["period_sq"] = wage_data["period"] ** 2

    # We only want to look at working age people
    wage_data = wage_data[wage_data["age"] < specs["max_ret_age"]]

    # partner data
    # wage_data["ln_partner_hrl_wage"] = np.log(wage_data["hourly_wage_p"])

    # prepare format
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    wage_data["constant"] = np.ones(len(wage_data))

    return wage_data


def calculate_partner_hours(path_dict):
    """Calculates average hours worked by working partners (i.e. conditional on working
    hours > 0)"""
    specs = read_and_derive_specs(path_dict["specs"])
    start_age = specs["start_age"]
    end_age = specs["max_ret_age"]
    # load data, filter, create age bins
    df = pd.read_pickle(
        path_dict["intermediate_data"] + "partner_wage_estimation_sample.pkl"
    )
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
