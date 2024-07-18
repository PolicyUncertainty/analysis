import numpy as np
import pandas as pd
import yaml


def generate_specs_and_update_params(path_dict, start_params):
    specs = generate_derived_and_data_derived_specs(path_dict)
    # Assign income shock scale to start_params_all
    start_params["sigma"] = specs["income_shock_scale"]
    return specs, start_params


def generate_derived_and_data_derived_specs(path_dict):
    specs = read_and_derive_specs(path_dict["specs"])

    # wages
    wage_params = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params.csv", index_col=0
    )

    specs["gamma_0"] = wage_params["constant"].values
    specs["gamma_1"] = wage_params["ln_exp"].values
    specs["income_shock_scale"] = wage_params["income_shock_std"].values.mean()

    # pensions
    specs["pension_point_value"] = 0.75 * specs["pension_point_value_west_2010"] + 0.25 * specs["pension_point_value_east_2010"] / specs["wealth_unit"]
    specs=pension_adjustment_factor(specs, path_dict)
    # max initial experience
    data_decision = pd.read_pickle(path_dict["intermediate_data"] + "structural_estimation_sample.pkl")
    specs["max_init_experience"] = (
        data_decision["experience"] - data_decision["period"]
    ).max()
    return specs


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


def pension_adjustment_factor(specs, path_dict):
    wage_params = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params.csv", index_col=0
    )
    wage_params_full_sample = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params_full_sample.csv", index_col=0
    )

    experience = np.arange(0, specs["exp_cap"] + 1)
    wage_by_experience_low = (
        np.exp(
            wage_params.loc[0, "constant"] + wage_params.loc[0,"ln_exp"] * np.log(experience + 1)
        )
    )
    wage_by_experience_high =  (
        np.exp(
            wage_params.loc[1, "constant"] + wage_params.loc[1,"ln_exp"] * np.log(experience + 1)
        )
    )
    wage_by_experience_average = (
        np.exp(
            wage_params_full_sample.loc["constant"].values + wage_params_full_sample.loc["ln_exp"].values * np.log(experience + 1)
        )
    )
    adjustment_factor_by_exp_low = wage_by_experience_low / wage_by_experience_average
    adjustment_factor_by_exp_high = wage_by_experience_high / wage_by_experience_average
    specs["adjustment_factor_by_exp_and_edu"] = np.array([adjustment_factor_by_exp_low, adjustment_factor_by_exp_high])
    return specs

