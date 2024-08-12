# Description: This file estimates the parameters of the MONTHLY wage equation using the SOEP panel data.
# We estimate the following equation for each education level:
# ln_partner_wage = beta_0 + beta_1 * ln(age) individual_FE + time_FE + epsilon

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS
from model_code.derive_specs import read_and_derive_specs
from estimation.first_step_estimation.est_wage_equation import prepare_estimation_data

def estimate_partner_wage_parameters(paths_dict):
    """Estimate the wage parameters for each education group in the sample."""
    # load and modify data
    wage_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_wage_estimation_sample.pkl"
    )
    wage_data = prepare_estimation_data(wage_data)
    breakpoint()
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
    wage_parameters.to_csv(paths_dict["est_results"] + "partner_wage_eq_params.csv")
    return coefficients