import numpy as np
import pandas as pd
from jax import numpy as jnp


def add_belief_process_specs(specs, path_dict):
    """Add both SRA and ERP belief parameters to specs."""
    # Add SRA belief parameters
    specs = add_sra_belief_specs(specs, path_dict)

    # Add ERP belief parameters (informed state transition parameters)
    specs = add_erp_belief_specs(specs, path_dict)

    return specs


def add_sra_belief_specs(specs, path_dict):
    """Add SRA belief parameters from new estimation results."""
    # Read SRA belief parameters from new location
    beliefs_params_df = pd.read_csv(
        path_dict["beliefs_est_results"] + "beliefs_parameters.csv"
    )

    # Extract alpha and sigma_sq parameters
    alpha_row = beliefs_params_df[beliefs_params_df["parameter"] == "alpha"]
    sigma_sq_row = beliefs_params_df[beliefs_params_df["parameter"] == "sigma_sq"]

    if len(alpha_row) == 0 or len(sigma_sq_row) == 0:
        raise ValueError(
            "Required SRA belief parameters (alpha, sigma_sq) not found in beliefs_parameters.csv"
        )

    specs["sra_belief_alpha"] = float(alpha_row["estimate"].iloc[0])
    specs["sra_belief_sigma_sq"] = float(sigma_sq_row["estimate"].iloc[0])

    return specs


def add_erp_belief_specs(specs, path_dict):
    """Add ERP belief parameters from beliefs_parameters.csv."""
    beliefs_params_df = pd.read_csv(
        path_dict["beliefs_est_results"] + "beliefs_parameters.csv"
    )

    n_edu_types = len(specs["education_labels"])
    informed_hazard_rate = np.zeros(n_edu_types)
    uninformed_penalties = np.zeros(n_edu_types)
    initial_shares = np.zeros(n_edu_types)

    # Fill arrays directly from CSV data
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        edu_params = beliefs_params_df[beliefs_params_df["type"] == edu_label]

        uninformed_penalties[edu_val] = edu_params[edu_params["parameter"] == "erp_uninformed_belief"]["estimate"].iloc[0] / 100
        informed_hazard_rate[edu_val] = edu_params[edu_params["parameter"] == "hazard_rate"]["estimate"].iloc[0]
        initial_shares[edu_val] = edu_params[edu_params["parameter"] == "initial_informed_share"]["estimate"].iloc[0]

    # Create age-indexed array (ages 0 to max_ret_age)
    ages_until_max_ret = np.arange(0, specs["max_ret_age"] + 1)
    informed_shares_in_ages = np.zeros((len(ages_until_max_ret), n_edu_types))

    # Fill informed shares by age (initial share is for age 17)
    for edu_val in range(n_edu_types):
        for age in ages_until_max_ret:
            if age < 17:
                # For ages below 17, assume 0 informed (no data)
                informed_shares_in_ages[age, edu_val] = 0
            else:
                # Calculate from age 17 using hazard rate model
                periods_since_17 = age - 17
                uninformed_prob = (1 - initial_shares[edu_val]) * (1 - informed_hazard_rate[edu_val]) ** periods_since_17
                informed_shares_in_ages[age, edu_val] = 1 - uninformed_prob

    specs["uninformed_ERP"] = uninformed_penalties
    specs["informed_hazard_rate"] = informed_hazard_rate
    specs["initial_informed_share"] = initial_shares
    specs["informed_shares_in_ages"] = informed_shares_in_ages
    return specs