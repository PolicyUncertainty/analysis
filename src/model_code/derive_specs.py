import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml


def generate_specs_and_update_params(path_dict, start_params):
    specs = generate_derived_and_data_derived_specs(path_dict)
    # Assign income shock scale to start_params_all
    start_params["sigma"] = specs["income_shock_scale"]
    start_params["interest_rate"] = specs["interest_rate"]
    start_params["beta"] = specs["discount_factor"]
    return specs, start_params


def generate_derived_and_data_derived_specs(path_dict, load_precomputed=False):
    specs = read_and_derive_specs(path_dict["specs"])

    # wages
    wage_params = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params.csv", index_col=0
    )

    specs["gamma_0"] = jnp.asarray(wage_params["constant"].values)
    specs["gamma_1"] = jnp.asarray(wage_params["ln_exp"].values)
    specs["income_shock_scale"] = wage_params["income_shock_std"].values.mean()

    # pensions
    specs["pension_point_value_by_edu_exp"] = calculate_pension_values(specs, path_dict)

    # Set initial experience
    specs["max_init_experience"] = create_initial_exp(path_dict, load_precomputed)
    return specs


def create_initial_exp(path_dict, load_precomputed):
    # Initial experience
    if load_precomputed:
        max_init_experience = int(
            np.loadtxt(path_dict["intermediate_data"] + "max_init_exp.txt")
        )
    else:
        # max initial experience
        data_decision = pd.read_pickle(
            path_dict["intermediate_data"] + "structural_estimation_sample.pkl"
        )
        max_init_experience = (
            data_decision["experience"] - data_decision["period"]
        ).max()
        np.savetxt(
            path_dict["intermediate_data"] + "max_init_exp.txt", [max_init_experience]
        )
    return max_init_experience


def read_and_derive_specs(spec_path):
    specs = yaml.safe_load(open(spec_path))

    # Number of periods in model
    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1
    # you can retire from min retirement age until max retirement age
    specs["n_possible_ret_ages"] = specs["max_ret_age"] - specs["min_ret_age"] + 1
    specs["n_policy_states"] = int(
        ((specs["max_SRA"] - specs["min_SRA"]) / specs["SRA_grid_size"]) + 1
    )
    specs["SRA_values_policy_states"] = np.arange(
        specs["min_SRA"],
        specs["max_SRA"] + specs["SRA_grid_size"],
        specs["SRA_grid_size"],
    )

    return specs


def calculate_pension_values(specs, path_dict):
    wage_params = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params.csv", index_col=0
    )
    wage_params_full_sample = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params_full_sample.csv", index_col=0
    )

    experience = np.arange(0, specs["exp_cap"] + 1)
    wage_by_experience_average = np.exp(
        wage_params_full_sample.loc["constant"].values
        + wage_params_full_sample.loc["ln_exp"].values * np.log(experience + 1)
    )
    # if number of education types changes, this needs to be adjusted
    wage_by_experience = np.ndarray(shape=(2, len(experience)))
    adjustment_factor_by_exp = np.ndarray(shape=(2, len(experience)))
    for education in [0, 1]:
        wage_by_experience[education] = np.exp(
            wage_params.loc[education, "constant"]
            + wage_params.loc[education, "ln_exp"] * np.log(experience + 1)
        )
        adjustment_factor_by_exp[education] = (
            wage_by_experience[education] / wage_by_experience_average
        )
        for i in range(1, len(experience)):
            adjustment_factor_by_exp[education, i] = adjustment_factor_by_exp[
                education, 1 : i + 1
            ].mean()

    # Generate average pension point value weighted by east and west
    # pensions
    pension_point_value = (
        0.75 * specs["pension_point_value_west_2010"]
        + 0.25 * specs["pension_point_value_east_2010"]
    ) / specs["wealth_unit"]
    return jnp.asarray(adjustment_factor_by_exp) * pension_point_value
