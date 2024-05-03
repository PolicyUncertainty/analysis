# Description: This file estimates the parameters of the MONTHLY wage equation using the SOEP panel data.
# We estimate the following equation:
# ln_wage = beta_0 + beta_1 * ln_(exp+1) + individual_FE + time_FE + epsilon
import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS
from model_code.derive_specs import read_and_derive_specs

def estimate_wage_parameters(paths, wage_data):

    coefficients = [0] * len(wage_data["education"].unique())
    breakpoint()
    wage_data["ln_wage"] = np.log(wage_data["wage"])
    wage_data["experience"] = wage_data["experience"] + 1
    wage_data["ln_exp"] = np.log(wage_data["experience"])

    # prepare estimation
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    wage_data["constant"] = np.ones(len(wage_data))
    for education in wage_data["education"].unique():
        wage_data = wage_data[wage_data["education"] == education]
        # estimate parametric regression, save parameters
        #breakpoint()
        model = PanelOLS(
            dependent=wage_data["ln_wage"],
            exog=wage_data[["constant", "ln_exp", "year"]],
            entity_effects=True,
            # time_effects=True,
        )
        fitted_model = model.fit(
            cov_type="clustered", cluster_entity=True, cluster_time=True
        )
        breakpoint()
        coefficients[education] = fitted_model.params[0:2]   
    return coefficients
    
