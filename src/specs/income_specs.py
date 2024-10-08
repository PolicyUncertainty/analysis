import numpy as np
import pandas as pd
from jax import numpy as jnp


def calculate_pension_values(specs, path_dict):
    wage_params = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params.csv", index_col=0
    )

    # Create possible experience values
    experience = np.arange(0, specs["exp_cap"] + 1)

    wage_by_experience_average = np.exp(
        wage_params.loc["all", "constant"]
        + wage_params.loc["all", "ln_exp"] * np.log(experience + 1)
    )
    mean_wage_all = wage_by_experience_average.mean()

    n_edu_types = specs["n_education_types"]
    # Create adjustment factor for pension point value container
    adjustment_factor_by_exp = np.ones(shape=(n_edu_types, len(experience)))

    for edu_index, edu_label in enumerate(specs["education_labels"]):
        for i in range(1, len(experience)):
            gamma_0 = wage_params.loc[edu_label, "constant"]
            gamma_1_plus_1 = wage_params.loc[edu_label, "ln_exp"] + 1
            mean_wage = (
                (np.exp(gamma_0) / gamma_1_plus_1) * ((i + 1) ** gamma_1_plus_1 - 1)
            ) / i

            adjustment_factor_by_exp[edu_index, i] = mean_wage / mean_wage_all

    # Generate average pension point value weighted by east and west
    # pensions
    pension_point_value = (
        0.75 * specs["pension_point_value_west_2010"]
        + 0.25 * specs["pension_point_value_east_2010"]
    ) / specs["wealth_unit"]
    return jnp.asarray(adjustment_factor_by_exp) * pension_point_value


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
