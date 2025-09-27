# Description: This file estimates the parameters of the partner wage equation using the SOEP panel data.
# We estimate the following equation for each education level:
# ln_partner_wage = beta_0 + beta_1 * ln(age) individual_FE + time_FE + epsilon
import numpy as np
import pandas as pd
import statsmodels.api as sm


def estimate_partner_wage_parameters(paths_dict, specs):
    """Estimate the wage parameters partners by education group in the sample.

    Parameters
    ----------
    paths_dict : dict
        Dictionary containing paths to data and output directories
    specs : dict
        Dictionary containing model specifications

    Returns
    -------
    None
        Results are saved to CSV files and wage data with predictions is saved for plotting
    """
    edu_labels = specs["education_labels"]
    wage_data = prepare_estimation_data(paths_dict, specs)

    wage_data = sm.add_constant(wage_data)

    wage_data["ln_period"] = np.log(wage_data["period"] + 1)

    wage_data = create_deflate_factor(paths_dict, specs, wage_data)
    wage_data["wage_p"] /= wage_data["deflate_factor"]

    # Estimate partner pensions
    wage_data.loc[wage_data["public_pension_p"] < 1, "public_pension_p"] = np.nan
    wage_data["public_pension_p"] /= wage_data["deflate_factor"]
    wage_data.groupby(["sex", "education"])["public_pension_p"].mean().to_csv(
        paths_dict["first_step_incomes"] + "partner_pension.csv"
    )
    wage_data = wage_data[wage_data["partner_state"] == 1]

    # Store all wage data with predictions for plotting
    all_wage_data_with_predictions = []

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        wage_data_sex = wage_data[wage_data["sex"] == sex_var]

        # Use cubic term for partner of men because of lfc labor supply pattern of women
        if sex_var == 0:
            covs = ["constant", "period", "period_sq", "period_cub"]
        else:
            covs = ["constant", "period", "period_sq"]

        # Initialize empty container for coefficients
        wage_parameters = pd.DataFrame(
            index=pd.Index(data=edu_labels, name="education"),
            columns=covs,
        )

        for edu_val, edu_label in enumerate(edu_labels):
            # Filter df
            wage_data_edu = wage_data_sex[wage_data_sex["education"] == edu_val].copy()

            if len(wage_data_edu) == 0:
                continue

            # make ols regression
            model = sm.OLS(
                endog=wage_data_edu["wage_p"],
                exog=wage_data_edu[covs].astype(float),
                missing="drop",
            )

            fitted_model = model.fit()
            print(fitted_model.summary())
            # Assign prediction
            wage_data_edu["wage_pred"] = fitted_model.predict()

            # Assign estimated parameters (column list corresponds to model params, so only these are assigned)
            wage_parameters.loc[edu_label] = fitted_model.params

            # Store this data for plotting
            all_wage_data_with_predictions.append(wage_data_edu)

        append = "men" if sex_var == 0 else "women"
        out_file_path = (
            paths_dict["first_step_incomes"] + f"partner_wage_eq_params_{append}.csv"
        )
        wage_parameters.to_csv(out_file_path)

    # Combine all wage data with predictions for plotting
    if all_wage_data_with_predictions:
        combined_wage_data = pd.concat(
            all_wage_data_with_predictions, ignore_index=True
        )
        # Save wage data with predictions for plotting
        combined_wage_data.to_csv(
            paths_dict["first_step_data"]
            + "partner_wage_estimation_sample_with_predictions.csv"
        )


def create_deflate_factor(paths_dict, specs, df):
    # Deflate wages with FEs from wage estimation
    wage_fe = pd.read_csv(
        paths_dict["first_step_incomes"] + "wage_eq_year_FE.csv", index_col=[0, 1]
    )
    fe_all = wage_fe.loc[("all", "all"), :]
    # Set index to int
    fe_all.index = fe_all.index.astype(int)
    deflate_factor = np.exp(fe_all).rename("deflate_factor")
    # Reference year is missing. Fill up
    deflate_factor.loc[specs["reference_year"]] = 1.0
    df["deflate_factor"] = df["year"].map(deflate_factor)
    return df


def prepare_estimation_data(paths_dict, specs):
    """Prepare the data for the wage estimation."""
    # load and modify data
    wage_data = pd.read_csv(
        paths_dict["first_step_data"] + "partner_wage_estimation_sample.csv"
    )

    wage_data = wage_data[wage_data["age"] < specs["max_age_partner_working"]]

    # Add period
    wage_data["period"] = wage_data["age"] - specs["start_age"]
    wage_data["period_sq"] = wage_data["period"] ** 2
    wage_data["period_cub"] = wage_data["period"] ** 3

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
    from specs.derive_specs import read_and_derive_specs

    specs = read_and_derive_specs(path_dict["specs"])
    start_age = specs["start_age"]
    end_age = specs["max_ret_age"]

    # load data, filter, create age bins
    df = pd.read_pickle(
        path_dict["first_step_data"] + "partner_wage_estimation_sample.pkl"
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
    out_file_path = path_dict["first_step_incomes"] + f"partner_hours.csv"
    partner_hours.to_csv(out_file_path)
    return partner_hours
