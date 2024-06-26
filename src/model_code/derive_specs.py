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

    # Generate number of policy states between 67 and min_SRA
    wage_params = pd.read_csv(
        path_dict["est_results"] + "wage_eq_params.csv", index_col=0
    )
    specs["gamma_0"] = wage_params["constant"].values
    specs["gamma_1"] = wage_params["ln_exp"].values
    specs["income_shock_scale"] = wage_params["income_shock_std"].values.mean()

    # calculate value of pension point based on unweighted average wage over 40 years
    # of work
    specs["pension_point_value"] = 27.2 / specs["wealth_unit"]

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
