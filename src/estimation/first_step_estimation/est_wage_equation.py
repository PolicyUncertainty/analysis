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
    # load and modify data
    wage_data = prepare_estimation_data(paths_dict)

    edu_labels = specs["education_labels"] + ["all"]
    model_params = ["constant", "ln_exp"]
    # Initialize empty container for coefficients
    wage_parameters = pd.DataFrame()
    for edu_val, edu_label in enumerate(edu_labels):
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
            wage_parameters.loc[edu_label, param + "_std"] = fitted_model.std_errors[
                param
            ]

        # Get estimate for income shock std
        rss = fitted_model.resids.T @ fitted_model.resids
        n_minus_k = wage_data_edu.shape[0] - fitted_model.params.shape[0]
        income_shock_var = rss / n_minus_k
        income_shock_std = np.sqrt(income_shock_var)
        wage_parameters.loc[edu_label, "income_shock_std"] = income_shock_std
        wage_parameters.loc[edu_label, "income_shock_std_std"] = np.sqrt(
            (2 * income_shock_var**2) / n_minus_k
        )
    # Save results
    wage_parameters.to_csv(paths_dict["est_results"] + "wage_eq_params.csv")
    wage_parameters.T.to_latex(
        paths_dict["tables"] + "wage_eq_params.tex", float_format="%.4f"
    )
    # print wage equation
    for edu_val, edu_label in enumerate(edu_labels):
        print("Hourly wage equation: " + edu_label)
        print("ln(hrly_wage) = "+str(wage_parameters.loc[edu_label, "constant"])+" + "+str(wage_parameters.loc[edu_label, "ln_exp"])+" * ln(exp+1) + epsilon")
        hrly_wage_with_20_exp = np.exp(
            wage_parameters.loc[edu_label, "constant"]
            + wage_parameters.loc[edu_label, "ln_exp"] * np.log(20)
        )
        print("Example: hourly wage with 20 years of experience: " + str(hrly_wage_with_20_exp))
        print("Income shock std: " + str(wage_parameters.loc[edu_label, "income_shock_std"]))
        print("--------------------")
    return wage_parameters


def prepare_estimation_data(paths_dict):
    """Prepare the data for the wage estimation."""
    wage_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "wage_estimation_sample.pkl"
    )

    # wage data
    wage_data["ln_wage"] = np.log(wage_data["hourly_wage"])
    wage_data["experience"] = wage_data["experience"] + 1
    wage_data["ln_exp"] = np.log(wage_data["experience"])
    wage_data["ln_age"] = np.log(wage_data["age"])

    # prepare format
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    wage_data["constant"] = np.ones(len(wage_data))

    return wage_data
