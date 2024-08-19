# Description: This file estimates the parameters of the MONTHLY wage equation using the SOEP panel data.
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
    # load and modify data
    wage_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_wage_estimation_sample.pkl"
    )
    if est_men == True:
        wage_data = wage_data[wage_data["sex"] == 1]
        out_file_path = paths_dict["est_results"] + "partner_wage_eq_params_men.csv"
    else:
        wage_data = wage_data[wage_data["sex"] == 2]
        out_file_path = paths_dict["est_results"] + "partner_wage_eq_params_women.csv"   
    wage_data = prepare_estimation_data(wage_data)

    # Initialize empty container for coefficients
    coefficients = [0] * len(wage_data["education"].unique())
    for education in wage_data["education"].unique():
        wage_data_edu = wage_data[wage_data["education"] == education]
        # estimate parametric regression, save parameters
        model = PanelOLS(
            dependent=wage_data_edu["ln_partner_wage"],
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
    wage_parameters.to_csv(out_file_path)
    return coefficients


def prepare_estimation_data(wage_data):
    """Prepare the data for the wage estimation."""
    # wage data
    wage_data["ln_wage"] = np.log(wage_data["wage"])
    wage_data["ln_age"] = np.log(wage_data["age"])

    # partner data
    wage_data["ln_partner_wage"] = np.log(wage_data["wage_p"])

    # prepare format
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    wage_data["constant"] = np.ones(len(wage_data))

    return wage_data