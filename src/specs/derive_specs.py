import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from specs.family_specs import calculate_partner_incomes
from specs.family_specs import predict_children_by_state
from specs.family_specs import read_in_partner_transition_specs
from specs.income_specs import get_pension_point_value
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
    specs["ppv"] = get_pension_point_value(specs)

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
    specs["max_init_experience"], specs["max_experience"] = create_max_experience(
        path_dict, specs, load_precomputed
    )

    specs["job_sep_probs"] = jnp.asarray(
        np.loadtxt(path_dict["est_results"] + "job_sep_probs.csv", delimiter=",")
    )

    # Assign population averages for annual hours and mean wage
    pop_averages = pd.read_csv(
        path_dict["est_results"] + "population_averages_working.csv"
    ).iloc[0]
    specs["av_annual_hours_ft"] = jnp.array(
        [
            pop_averages["annual_hours_low_ft_work"],
            pop_averages["annual_hours_high_ft_work"],
        ]
    )
    specs["av_annual_hours_pt"] = jnp.array(
        [
            pop_averages["annual_hours_low_pt_work"],
            pop_averages["annual_hours_high_pt_work"],
        ]
    )
    specs["mean_wage"] = pop_averages["annual_mean_wage"]

    # read informed state transition parameters
    specs["informed_hazard_rate"] = pd.read_pickle(path_dict["est_results"] + "uninformed_hazard_rate.pkl")
    specs["uninformed_early_retirement_penalty"] = pd.read_pickle(path_dict["est_results"] + "uninformed_average_belief.pkl")/100
    return specs


def create_max_experience(path_dict, specs, load_precomputed):
    # Initial experience
    if load_precomputed:
        max_init_experience = int(
            np.loadtxt(path_dict["intermediate_data"] + "max_init_exp.txt")
        )
        max_experience = int(np.loadtxt(path_dict["intermediate_data"] + "max_exp.txt"))
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

        # Now max overall
        max_experience = data_decision["experience"].max()
        np.savetxt(path_dict["intermediate_data"] + "max_exp.txt", [max_experience])
    return max_init_experience, max_experience


def read_and_derive_specs(spec_path):
    specs = yaml.safe_load(open(spec_path))

    # Number of periods in model
    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1
    # Number of education types
    specs["n_education_types"] = len(specs["education_labels"])
    # you can retire from min retirement age until max retirement age
    specs["n_policy_states"] = (
        int(((specs["max_SRA"] - specs["min_SRA"]) / specs["SRA_grid_size"]) + 1) + 1
    )
    specs["SRA_values_policy_states"] = np.arange(
        specs["min_SRA"],
        specs["max_SRA"] + specs["SRA_grid_size"],
        specs["SRA_grid_size"],
    )
    return specs
