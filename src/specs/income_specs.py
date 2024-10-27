import numpy as np
import pandas as pd
from jax import numpy as jnp


def get_pension_vars(specs, path_dict):
    pop_averages = pd.read_csv(
        path_dict["est_results"] + "population_averages_working.csv", index_col=0
    )

    mean_wage_all = pop_averages["annual_mean_wage"]

    # Generate average pension point value weighted by east and west
    # pensions
    pension_point_value = (
        0.75 * specs["pension_point_value_west_2010"]
        + 0.25 * specs["pension_point_value_east_2010"]
    ) / specs["wealth_unit"]
    return pension_point_value, mean_wage_all


def process_wage_params(path_dict, specs):
    # wages
    wage_params = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params.csv", index_col=0
    )
    edu_labels = specs["education_labels"]

    gamma_0 = jnp.asarray(wage_params.loc[edu_labels, "constant"].values)
    gamma_1 = jnp.asarray(wage_params.loc[edu_labels, "ln_exp"].values)
    income_shock_scale = wage_params.loc[edu_labels, "income_shock_std"].values.mean()
    return gamma_0, gamma_1, income_shock_scale
