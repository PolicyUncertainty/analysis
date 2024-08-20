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
        sex_var = 1
    else:
        sex_var = 2

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
