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
    model_params = ["constant", "ln_exp"]
    # Initialize empty container for coefficients
    wage_parameters = pd.DataFrame()

    # Initialize years for year fixed effects
    year_fixed_effects = {}
    years = list(range(specs["start_year"] + 1, specs["end_year"] + 1))
    for edu_val, edu_label in enumerate(edu_labels):
        year_fixed_effects[edu_label] = {}
        if edu_label == "all":
            wage_data_edu = wage_data
        else:
            wage_data_edu = wage_data[wage_data["education"] == edu_val]

        # estimate parametric regression, save parameters
        model = PanelOLS(
            dependent=wage_data_edu["ln_wage"],
            exog=wage_data_edu[model_params + ["year"]],
            entity_effects=True,
        )
        fitted_model = model.fit(
            cov_type="clustered", cluster_entity=True, cluster_time=True
        )
        # Assign estimated parameters (column list corresponds to model params, so only these are assigned)
        for param in model_params:
            wage_parameters.loc[edu_label, param] = fitted_model.params[param]
            wage_parameters.loc[edu_label, param + "_ser"] = fitted_model.std_errors[
                param
            ]

        for year in years:
            year_fixed_effects[edu_label][year] = fitted_model.params[f"year.{year}"]

        # Get estimate for income shock std
        (
            wage_parameters.loc[edu_label, "income_shock_std"],
            wage_parameters.loc[edu_label, "income_shock_std_ser"],
        ) = est_shock_std(
            residuals=fitted_model.resids,
            n_obs=wage_data_edu.shape[0],
            n_params=fitted_model.params.shape[0],
        )
    # Save results
    wage_parameters.to_csv(paths_dict["est_results"] + "wage_eq_params.csv")
    wage_parameters.T.to_latex(
        paths_dict["tables"] + "wage_eq_params.tex", float_format="%.4f"
    )
    # After estimation print some summary statistics
    print_wage_equation(wage_parameters, edu_labels)

    calc_additional_wage_params(wage_data, year_fixed_effects, specs, paths_dict)

    return wage_parameters


def calc_additional_wage_params(df, year_fixed_effects, specs, paths_dict):
    """Save population average of annual wage (for pension calculation) and working
    hours by education (to compute annual wages).

    We do this here (as opposed to model specs) to avoid loading the data twice.

    """
    years = list(range(specs["start_year"] + 1, specs["end_year"] + 1))
    edu_labels = specs["education_labels"]

    # # Now use results to deflate wages to 2010 levels to calculate some population statistics
    df["ln_wage_deflated"] = df["ln_wage"].copy()
    for edu_val, edu_label in enumerate(edu_labels):
        for year in years:
            edu_mask = df["education"] == edu_val
            year_mask = df["year"] = year
            df.loc[edu_mask & year_mask, "ln_wage_deflated"] -= year_fixed_effects[
                edu_label
            ][year]

    df["annual_wage_deflated"] = (
        np.exp(df["ln_wage_deflated"]) * df["monthly_hours"]
    ) * 12
    pop_avg_annual_wage = df["annual_wage_deflated"].mean()

    df["yearly_hours"] = df["monthly_hours"] * 12
    avg_hours_by_edu_choice = df.groupby(["education", "choice"])["yearly_hours"].mean()

    choice_mapping = {"pt_work": 2, "ft_work": 3}

    edu_labels_params = ["low", "high"]
    pop_avg = pd.DataFrame({"annual_mean_wage": [pop_avg_annual_wage]})
    for choice, choice_var in choice_mapping.items():
        for edu_val, edu_label in enumerate(edu_labels_params):
            pop_avg[f"annual_hours_{edu_label}_{choice}"] = avg_hours_by_edu_choice.loc[
                (edu_val, choice_var)
            ]

    pop_avg.to_csv(
        paths_dict["est_results"] + "population_averages_working.csv", index=False
    )
    return pop_avg


def est_shock_std(residuals, n_obs, n_params):
    """Estimate income shock std and its standard error."""
    rss = residuals @ residuals
    n_minus_k = n_obs - n_params
    income_shock_var = rss / n_minus_k
    income_shock_std = np.sqrt(income_shock_var)
    income_shock_std_ser = np.sqrt((2 * income_shock_var**2) / n_minus_k)
    return income_shock_std, income_shock_std_ser


def print_wage_equation(wage_parameters, edu_labels):
    # print wage equation
    for edu_val, edu_label in enumerate(edu_labels):
        print("Hourly wage equation: " + edu_label)
        print(
            "ln(hrly_wage) = "
            + str(wage_parameters.loc[edu_label, "constant"])
            + " + "
            + str(wage_parameters.loc[edu_label, "ln_exp"])
            + " * ln(exp+1) + epsilon"
        )
        hrly_wage_with_20_exp = np.exp(
            wage_parameters.loc[edu_label, "constant"]
            + wage_parameters.loc[edu_label, "ln_exp"] * np.log(20)
        )
        print(
            "Example: hourly wage with 20 years of experience: "
            + str(hrly_wage_with_20_exp)
        )
        print(
            "Income shock std: "
            + str(wage_parameters.loc[edu_label, "income_shock_std"])
        )
        print("--------------------")
