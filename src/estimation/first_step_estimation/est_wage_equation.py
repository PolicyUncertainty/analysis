# Description: This file estimates the parameters of the MONTHLY wage equation using the SOEP data.
# We estimate the following equation:
# wage = beta_0 + beta_1 * experience + beta_2 * experience^2 + individual_FE + time_FE + epsilon
import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS
from model_code.derive_specs import read_and_derive_specs


def estimate_wage_parameters(paths, wage_data):
    specs = read_and_derive_specs(paths["specs"])

    # wage truncation
    truncation_percentiles = [
    specs["wage_trunc_low_perc"],
    specs["wage_trunc_high_perc"],
    ]
    wage_percentiles = wage_data["experience"].quantile(truncation_percentiles)
    wage_data = wage_data[
        (wage_data["experience"] >= wage_percentiles.iloc[0])
        & (wage_data["wage"] <= wage_percentiles.iloc[1])
    ]

    # prepare estimation
    wage_data["year"] = wage_data["syear"].astype("category")
    wage_data = wage_data.set_index(["pid", "syear"])
    
    wage_data["experience_sq"] = wage_data["experience"] ** 2
    wage_data["constant"] = np.ones(len(wage_data))

    # estimate parametric regression, save parameters
    model = PanelOLS(
        dependent=wage_data["wage"] / specs["wealth_unit"],
        exog=wage_data[["constant", "experience", "experience_sq", "year"]],
        entity_effects=True,
        # time_effects=True,
    )
    fitted_model = model.fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    coefficients = fitted_model.params[0:3]

    # model.fit().std_errors.
    # TODO: fix degrees of freedom hardcoding
    coefficients.loc["income_shock_std"] = np.sqrt(
        model.fit().resid_ss / (wage_data.shape[0] - 14763)
    )

    print("Estimated wage equation coefficients:\n{}".format(coefficients.to_string()))

    # Export regression coefficients
    coefficients.to_csv(paths["est_results"] + "wage_eq_params.csv")
    return coefficients