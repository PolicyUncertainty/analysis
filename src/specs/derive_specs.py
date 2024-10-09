import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from specs.family_specs import calculate_partner_incomes
from specs.family_specs import predict_children_by_state
from specs.family_specs import read_in_partner_transition_specs
from specs.income_specs import calculate_pension_values
from specs.income_specs import process_wage_params


def generate_derived_and_data_derived_specs(path_dict, load_precomputed=False):
    """This function reads in specs and adds derived and data estimated specs."""
    specs = read_and_derive_specs(path_dict["specs"])

    # wages
    (
        specs["gamma_0"],
        specs["gamma_1"],
        specs["income_shock_scale"],
    ) = process_wage_params(path_dict, specs)

    # pensions
    specs["ppv"], specs["mean_wage"] = calculate_pension_values(specs, path_dict)

    # partner income
    specs["partner_wage"], specs["partner_pension"] = calculate_partner_incomes(
        path_dict, specs
    )
    # specs["partner_hours"] = calculate_partner_hours(path_dict, specs)
    # specs["partner_pension"] = calculate_partner_pension(path_dict)

    # family transitions
    specs["children_by_state"] = predict_children_by_state(path_dict, specs)

    # Read in family transitions
    (
        specs["partner_trans_mat"],
        specs["n_partner_states"],
    ) = read_in_partner_transition_specs(path_dict, specs)

    # Set initial experience
    specs["max_init_experience"], specs["experience_grid"] = create_model_initials(
        path_dict, specs, load_precomputed
    )

    specs["job_sep_probs"] = jnp.asarray(
        np.loadtxt(path_dict["est_results"] + "job_sep_probs.csv", delimiter=",")
    )
    return specs


def create_model_initials(path_dict, specs, load_precomputed):
    # Initial experience
    if load_precomputed:
        max_init_experience = int(
            np.loadtxt(path_dict["intermediate_data"] + "max_init_exp.txt")
        )
        experience_grid = np.loadtxt(
            path_dict["intermediate_data"] + "experience_grid.txt"
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
        # experience grid
        max_period_exp = max_init_experience + data_decision["period"]
        data_decision["exp_share"] = data_decision["experience"].values / max_period_exp
        experience_quantile_grid = np.linspace(0, 1, specs["n_experience_grid_points"])
        experience_grid = (
            data_decision["exp_share"].quantile(experience_quantile_grid).values
        )
        np.savetxt(
            path_dict["intermediate_data"] + "experience_grid.txt", experience_grid
        )
    return max_init_experience, jnp.asarray(experience_grid)


def read_and_derive_specs(spec_path):
    specs = yaml.safe_load(open(spec_path))

    # Number of periods in model
    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1
    # Number of education types
    specs["n_education_types"] = len(specs["education_labels"])
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
