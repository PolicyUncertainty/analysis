# Description: This file estimates the parameters of the MONTHLY wage equation using the SOEP panel data.
# We estimate the following equation:
# ln_wage = beta_0 + beta_1 * ln_(exp+1) + individual_FE + time_FE + epsilon
import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS
from model_code.derive_specs import read_and_derive_specs
import matplotlib.pyplot as plt


def estimate_wage_parameters(wage_data):

    coefficients = [0] * len(wage_data["education"].unique())
    wage_data["ln_wage"] = np.log(wage_data["wage"])
    wage_data["experience"] = wage_data["experience"] + 1
    wage_data["ln_exp"] = np.log(wage_data["experience"])

    # prepare estimation
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    wage_data["constant"] = np.ones(len(wage_data))
    for education in wage_data["education"].unique():
        wage_data_edu = wage_data[wage_data["education"] == education]
        # estimate parametric regression, save parameters
        model = PanelOLS(
            dependent=wage_data_edu["ln_wage"],
            exog=wage_data_edu[["constant", "ln_exp", "year"]],
            entity_effects=True,
            # time_effects=True,
        )
        fitted_model = model.fit(
            cov_type="clustered", cluster_entity=True, cluster_time=True
        )
        coefficients[education] = list(fitted_model.params[0:2])
        income_shock_std = np.sqrt(fitted_model.resids.var())
        coefficients[education] = coefficients[education] + [income_shock_std]
    return coefficients
    
# visualize wages by experience

def plot_wages_by_edu(coefficients):
    fig, ax = plt.subplots()
    for i, coef in enumerate(coefficients):
        wage = np.exp(coef[0] + coef[1] * np.log(range(1, 46)))
        ax.plot(range(1, 46), wage, label=i)
    ax.legend()
    plt.show()