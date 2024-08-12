# Description: This file estimates the parameters of the MONTHLY wage equation using the SOEP panel data.
# We estimate the following equation for each education level:
# ln_wage = beta_0 + beta_1 * ln_(exp+1) + individual_FE + time_FE + epsilon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS
from model_code.derive_specs import read_and_derive_specs


def estimate_wage_parameters(paths_dict):
    """Estimate the wage parameters for each education group in the sample."""    
    # load and modify data
    wage_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "wage_estimation_sample.pkl"
    )
    wage_data = prepare_estimation_data(wage_data)
    
    # Initialize empty container for coefficients
    coefficients = [0] * len(wage_data["education"].unique())
    for education in wage_data["education"].unique():
        wage_data_edu = wage_data[wage_data["education"] == education]
        # estimate parametric regression, save parameters
        model = PanelOLS(
            dependent=wage_data_edu["ln_wage"],
            exog=wage_data_edu[["constant", "ln_exp", "year"]],
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
    wage_parameters.to_csv(paths_dict["est_results"] + "wage_eq_params.csv")
    return coefficients


def estimate_average_wage_parameters(paths_dict):
    """Estimate the average wage parameters for all individuals in the sample."""
    wage_data = prepare_estimation_data(paths_dict)

    # estimate parametric regression, save parameters
    model = PanelOLS(
        dependent=wage_data["ln_wage"],
        exog=wage_data[["constant", "ln_exp", "year"]],
        entity_effects=True,
    )
    fitted_model = model.fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    coefficients = fitted_model.params[0:2]
    income_shock_std = np.sqrt(fitted_model.resids.var())
    coefficients.loc["income_shock_std"] = income_shock_std
    coefficients.name = "all"

    wage_parameters = pd.DataFrame(coefficients)
    wage_parameters.to_csv(paths_dict["est_results"] + "wage_eq_params_full_sample.csv")
    return coefficients


def prepare_estimation_data(wage_data):
    """Prepare the data for the wage estimation."""
    # wage data
    wage_data["ln_wage"] = np.log(wage_data["wage"])
    wage_data["experience"] = wage_data["experience"] + 1
    wage_data["ln_exp"] = np.log(wage_data["experience"])
    wage_data["ln_age"] = np.log(wage_data["age"])

    # partner data
    try: wage_data["ln_partner_wage"] = np.log(wage_data["wage_p"])
    except KeyError: pass

    # prepare format
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    wage_data["constant"] = np.ones(len(wage_data))

    return wage_data
