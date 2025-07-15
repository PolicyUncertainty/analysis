# Description: This file estimates the parameters of the HOURLY wage equation using the SOEP panel data.
# We estimate the following equation for each education level:
# ln_partner_wage = beta_0 + beta_1 * ln(age) individual_FE + time_FE + epsilon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from export_results.figures.color_map import JET_COLOR_MAP
from specs.derive_specs import read_and_derive_specs


def estimate_partner_wage_parameters(paths_dict, specs):
    """Estimate the wage parameters partners by education group in the sample.

    Est_men is a boolean that determines whether the estimation is done for men or for
    women.

    """
    edu_labels = specs["education_labels"]
    covs = ["constant", "period", "period_sq"]
    wage_data = prepare_estimation_data(paths_dict, specs)
    wage_data = sm.add_constant(wage_data)

    # Deflate wages with FEs from wage estimaten
    wage_fe = pd.read_csv(
        paths_dict["est_results"] + "wage_eq_year_FE.csv", index_col=[0, 1]
    )
    fe_all = wage_fe.loc[("all", "all"), :]
    # Set index to int
    fe_all.index = fe_all.index.astype(int)
    deflate_factor = np.exp(fe_all).rename("deflate_factor")
    # Reference year is missing. Fill up
    deflate_factor.loc[specs["reference_year"]] = 1.0
    wage_data["deflate_factor"] = wage_data["year"].map(deflate_factor)
    wage_data["wage_p"] /= wage_data["deflate_factor"]

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        wage_data_sex = wage_data[wage_data["sex"] == sex_var]

        # Initialize empty container for coefficients
        wage_parameters = pd.DataFrame(
            index=pd.Index(data=edu_labels, name="education"),
            columns=covs,
        )
        fig, ax = plt.subplots()
        for edu_val, edu_label in enumerate(edu_labels):
            # Filter df
            wage_data_edu = wage_data_sex[wage_data_sex["education"] == edu_val].copy()

            # make ols regression
            model = sm.OLS(
                endog=wage_data_edu["wage_p"],
                exog=wage_data_edu[covs].astype(float),
                missing="drop",
            )

            fitted_model = model.fit()
            # Assign prediction
            wage_data_edu["wage_pred"] = fitted_model.predict()
            # Plot wage and prediction
            ax.plot(
                wage_data_edu.groupby("age")["wage_p"].mean(),
                label=f"Obs. {edu_label}",
                linestyle="--",
                color=JET_COLOR_MAP[edu_val],
            )
            ax.plot(
                wage_data_edu.groupby("age")["wage_pred"].mean(),
                label=f"Est. {edu_label}",
                color=JET_COLOR_MAP[edu_val],
            )

            # Assign estimated parameters (column list corresponds to model params, so only these are assigned)
            wage_parameters.loc[edu_label] = fitted_model.params

        append = "men" if sex_var == 0 else "women"
        ax.legend()
        ax.set_title(f"Partner Wages of {sex_label}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Monthly Wage")
        fig.savefig(paths_dict["plots"] + f"partner_wages_{append}.png")

        out_file_path = (
            paths_dict["est_results"] + f"partner_wage_eq_params_{append}.csv"
        )
        wage_parameters.to_csv(out_file_path)


def prepare_estimation_data(paths_dict, specs):
    """Prepare the data for the wage estimation."""
    # load and modify data
    wage_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_wage_estimation_sample.pkl"
    )

    # Add period
    wage_data["period"] = wage_data["age"] - specs["start_age"]
    wage_data["period_sq"] = wage_data["period"] ** 2

    # We only want to look at working age people
    wage_data = wage_data[wage_data["age"] < specs["max_ret_age"]]

    # prepare format
    wage_data["year"] = wage_data["syear"].astype("int")
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
