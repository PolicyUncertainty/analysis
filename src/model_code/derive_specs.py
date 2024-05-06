import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from model_code.wealth_and_budget.main_budget_equation import calc_deduction


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

    specs["gamma_0"] = wage_params.loc["constant", "parameter"]
    specs["gamma_1"] = wage_params.loc["full_time_exp", "parameter"]
    specs["gamma_2"] = wage_params.loc["full_time_exp_sq", "parameter"]
    specs["income_shock_scale"] = wage_params.loc["income_shock_std", "parameter"]

    # calculate value of pension point based on unweighted average wage over 40 years
    # of work
    specs["pension_point_value"] = 27.2 / specs["wealth_unit"]

    # max initial experience
    data_decision = pd.read_pickle(path_dict["intermediate_data"] + "decision_data.pkl")
    specs["max_init_experience"] = (
        data_decision["experience"] - data_decision["period"]
    ).max()
    return specs


def read_and_derive_specs(spec_path):
    specs = yaml.safe_load(open(spec_path))

    # Number of periods in model
    specs["n_periods_main"] = specs["max_ret_age"] - specs["start_age"] + 1
    specs["n_periods_old_age"] = specs["end_age"] - specs["max_ret_age"] + 1
    # you can retire from min retirement age until max retirement age
    specs["n_possible_ret_ages"] = specs["max_ret_age"] - specs["min_ret_age"] + 1
    specs["n_policy_states"] = int(
        ((specs["max_SRA"] - specs["min_SRA"]) / specs["SRA_grid_size"]) + 1
    )
    # Generate possible deduction states for the old age problem from all possible
    # retirement ages and policy states.
    deductions_all = []
    for policy_state in range(specs["n_policy_states"]):
        for retirement_age_id in range(specs["n_possible_ret_ages"]):
            deduction = calc_deduction(policy_state, retirement_age_id, specs)
            deductions_all += [deduction]

    unique_deduct_values, index_mapping = np.unique(
        np.array(deductions_all), return_inverse=True
    )
    specs["deduction_state_values"] = unique_deduct_values
    specs["n_deduction_states"] = unique_deduct_values.shape[0]
    specs["old_age_state_index_mapping"] = jnp.array(
        index_mapping.reshape((specs["n_policy_states"], specs["n_possible_ret_ages"]))
    )

    specs["SRA_values_policy_states"] = np.arange(
        specs["min_SRA"],
        specs["max_SRA"] + specs["SRA_grid_size"],
        specs["SRA_grid_size"],
    )

    return specs
