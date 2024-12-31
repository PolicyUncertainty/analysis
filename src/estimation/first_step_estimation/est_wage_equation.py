# Description: This file estimates the parameters of the HOURLY wage equation using the SOEP panel data.
# We estimate the following equation for each education level:
# ln_wage = beta_0 + beta_1 * ln_(exp+1) + individual_FE + time_FE + epsilon
import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS


def estimate_wage_parameters(paths_dict, specs):
    """Estimate the wage parameters for each education group in the sample.

    Also estimate for all individuals.

    """
    # Load and modify data.
    wage_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "wage_estimation_sample.pkl"
    )

    # Hourly wage
    wage_data["ln_wage"] = np.log(wage_data["hourly_wage"])

    # Log exp and other explanatory variablesq
    wage_data["ln_exp"] = np.log(wage_data["experience"] + 1)
    wage_data["constant"] = np.ones(len(wage_data))

    # prepare format
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])

    edu_labels = specs["education_labels"] + ["all"]
    sex_labels = specs["sex_labels"] 
    model_params = ["constant", "ln_exp"]
    # Initialize empty container for coefficients
    index = pd.MultiIndex.from_product([edu_labels, sex_labels, model_params + [param + "_ser" for param in model_params]], names=["education", "sex", "parameter"])    
    wage_parameters = pd.DataFrame(index=index, columns=["value"])

    # Initialize years for year fixed effects
    year_fixed_effects = {}
    years = list(range(specs["start_year"] + 1, specs["end_year"] + 1))
    for edu_val, edu_label in enumerate(edu_labels):
        for sex_val, sex_label in enumerate(sex_labels):
            year_fixed_effects[edu_label, sex_label] = {}
            if edu_label == "all":
                wage_data_type = wage_data
            else:
                wage_data_type = wage_data[(wage_data["education"] == edu_val) & (wage_data["sex"] == sex_val)]

            # estimate parametric regression, save parameters
            model = PanelOLS(
                dependent=wage_data_type["ln_wage"],
                exog=wage_data_type[model_params + ["year"]],
                entity_effects=True,
            )
            fitted_model = model.fit(
                cov_type="clustered", cluster_entity=True, cluster_time=True
            )

            # Assign estimated parameters (column list corresponds to model params, so only these are assigned)
            for param in model_params:
                wage_parameters.loc[edu_label, sex_label, param] = fitted_model.params[param]
                wage_parameters.loc[edu_label, sex_label, param + "_ser"] = fitted_model.std_errors[
                    param
                ]
            for year in years:
                year_fixed_effects[(edu_label, sex_label)][year] = fitted_model.params[f"year.{year}"]

            # Get estimate for income shock std
            (
                wage_parameters.loc[edu_label, sex_label, "income_shock_std"],
                wage_parameters.loc[edu_label, sex_label, "income_shock_std_ser"],
            ) = est_shock_std(
                residuals=fitted_model.resids,
                n_obs=wage_data_type.shape[0],
                n_params=fitted_model.params.shape[0],
            )
    # Save results
    wage_parameters.to_csv(paths_dict["est_results"] + "wage_eq_params.csv")
    wage_parameters.T.to_latex(
        paths_dict["tables"] + "wage_eq_params.tex", float_format="%.4f"
    )
    # After estimation print some summary statistics
    print_wage_equation(wage_parameters, edu_labels, sex_labels)

    calc_additional_wage_params(wage_data, year_fixed_effects, specs, paths_dict)

    return wage_parameters


def calc_additional_wage_params(df, year_fixed_effects, specs, paths_dict):
    """Save population average of annual wage (for pension calculation) and working
    hours by education (to compute annual wages).

    We do this here (as opposed to model specs) to avoid loading the data twice.

    """
    years = list(range(specs["start_year"] + 1, specs["end_year"] + 1))
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]

    # # Now use results to deflate wages to 2010 levels to calculate population statistics
    df["ln_wage_deflated"] = df["ln_wage"].copy()
    for edu_val, edu_label in enumerate(edu_labels):
        for sex_val, sex_label in enumerate(sex_labels):
            for year in years:
                edu_mask = df["education"] == edu_val
                sex_mask = df["sex"] == sex_val
                year_mask = df["year"] = year
                df.loc[edu_mask & sex_mask & year_mask, "ln_wage_deflated"] -= year_fixed_effects[
                    (edu_label, sex_label)
                ][year]

    df["annual_hours"] = df["monthly_hours"] * 12

    df["annual_wage_deflated"] = np.exp(df["ln_wage_deflated"]) * df["annual_hours"]
    pop_avg_annual_wage = df["annual_wage_deflated"].mean()

    avg_hours_by_type_choice = df.groupby(["education", "sex", "choice"])["annual_hours"].mean()

    avg_hours_by_type_choice.to_csv(
        paths_dict["est_results"] + "population_averages_working.csv", index=True
    )
    print(
        "Population averages for working hours: \n"
    )
    print(avg_hours_by_type_choice)

    return avg_hours_by_type_choice


def est_shock_std(residuals, n_obs, n_params):
    """Estimate income shock std and its standard error."""
    rss = residuals @ residuals
    n_minus_k = n_obs - n_params
    income_shock_var = rss / n_minus_k
    income_shock_std = np.sqrt(income_shock_var)
    income_shock_std_ser = np.sqrt((2 * income_shock_var**2) / n_minus_k)
    return income_shock_std, income_shock_std_ser


def print_wage_equation(wage_parameters, edu_labels, sex_labels):
    # print wage equation
    for edu_val, edu_label in enumerate(edu_labels):
        for sex_val, sex_label in enumerate(sex_labels):
            print("Hourly wage equation: " + edu_label + " " + sex_label)
            print(
                "ln(hrly_wage) = "
                + str(wage_parameters.loc[(edu_label, sex_label, "constant"), "value"])
                + " + "
                + str(wage_parameters.loc[(edu_label, sex_label, "ln_exp"), "value"])
                + " * ln(exp+1) + epsilon"
            )
            hrly_wage_with_20_exp = np.exp(
                wage_parameters.loc[(edu_label, sex_label, "constant"), "value"]
                + wage_parameters.loc[(edu_label, sex_label, "ln_exp"), "value"] * np.log(20)
            )
            print(
                "Example: hourly wage with 20 years of experience: "
                + str(hrly_wage_with_20_exp)
            )
            print(
                "Income shock std: "
                + str(wage_parameters.loc[edu_label, sex_label, "income_shock_std"])
            )
            print("--------------------")
