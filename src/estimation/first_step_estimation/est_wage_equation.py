# Description: This file estimates the parameters of the MONTHLY wage equation using the SOEP panel data.
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
    wage_parameters = pd.DataFrame(
        index=pd.Index(data=edu_labels, name="education"),
        columns=model_params + ["income_shock_std"],
    )
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
        wage_parameters.loc[edu_label] = fitted_model.params

        # Get estimate for income shock std
        income_shock_std = np.sqrt(fitted_model.resids.var())
        wage_parameters.loc[edu_label, "income_shock_std"] = income_shock_std

    wage_parameters.to_csv(paths_dict["est_results"] + "wage_eq_params.csv")
    return wage_parameters


def prepare_estimation_data(paths_dict):
    """Prepare the data for the wage estimation."""
    wage_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "wage_estimation_sample.pkl"
    )

    # wage data
    wage_data["ln_wage"] = np.log(wage_data["wage"])
    wage_data["experience"] = wage_data["experience"] + 1
    wage_data["ln_exp"] = np.log(wage_data["experience"])
    wage_data["ln_age"] = np.log(wage_data["age"])

    # prepare format
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    wage_data["constant"] = np.ones(len(wage_data))

    return wage_data
