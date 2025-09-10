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
        raise ValueError("Required SRA belief parameters (alpha, sigma_sq) not found in beliefs_parameters.csv")
    
    specs["sra_belief_alpha"] = float(alpha_row["estimate"].iloc[0])
    specs["sra_belief_sigma_sq"] = float(sigma_sq_row["estimate"].iloc[0])
    
    return specs


def add_erp_belief_specs(specs, path_dict):
    """Add ERP belief parameters from beliefs_parameters.csv."""
    # Read structured belief parameters
    beliefs_params_df = pd.read_csv(
        path_dict["beliefs_est_results"] + "beliefs_parameters.csv"
    )
    
    # Extract ERP parameters
    erp_params = beliefs_params_df[
        beliefs_params_df["parameter"].isin(["initial_informed_share", "hazard_rate"])
    ]
    
    # Create hazard rate DataFrame in expected format
    df_informed_hazard_rate = pd.DataFrame(
        index=["initial_informed_share", "hazard_rate"],
        columns=specs["education_labels"]
    )
    
    # Fill from structured data
    for _, row in erp_params.iterrows():
        param_name = row["parameter"]
        edu_label = row["type"]  # education label is in 'type' column
        estimate = row["estimate"]
        df_informed_hazard_rate.loc[param_name, edu_label] = estimate
    
    # Create placeholder DataFrames for uninformed penalties and predicted shares
    # These are currently not used in the main model, but kept for compatibility
    df_uninformed_penalties = pd.DataFrame(
        index=["erp_uninformed_belief", "erp_uninformed_belief_sem"],
        columns=specs["education_labels"]
    ).fillna(0)  # Fill with zeros as placeholder
    
    df_predicted_shares = pd.DataFrame(columns=specs["education_labels"])
    n_edu_types = len(specs["education_labels"])

    informed_hazard_rate = np.zeros(n_edu_types, dtype=float)
    uninformed_penalties = np.zeros(n_edu_types, dtype=float)
    initial_shares = np.zeros(n_edu_types, dtype=float)

    # Get working ages and also number of working ages
    ages_until_max_ret = np.arange(0, specs["max_ret_age"] + 1)
    informed_shares_in_ages = np.zeros(
        (len(ages_until_max_ret), n_edu_types), dtype=float
    )
    
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        # Extract values from DataFrame, handling potential missing values
        uninformed_penalties[edu_val] = (
            df_uninformed_penalties.loc["erp_uninformed_belief", edu_label] / 100
            if edu_label in df_uninformed_penalties.columns else 0
        )
        informed_hazard_rate[edu_val] = (
            df_informed_hazard_rate.loc["hazard_rate", edu_label]
            if edu_label in df_informed_hazard_rate.columns else 0
        )
        initial_shares[edu_val] = (
            df_informed_hazard_rate.loc["initial_informed_share", edu_label]
            if edu_label in df_informed_hazard_rate.columns else 0
        )
        
        # Handle predicted shares if available
        if edu_label in df_predicted_shares.columns and len(df_predicted_shares) > 0:
            predicted_ages = df_predicted_shares.index.values
            # Only assign if ages are within range
            valid_ages = predicted_ages[predicted_ages < len(informed_shares_in_ages)]
            informed_shares_in_ages[valid_ages, edu_val] = df_predicted_shares.loc[
                valid_ages, edu_label
            ]

    specs["uninformed_ERP"] = jnp.asarray(uninformed_penalties)
    specs["informed_hazard_rate"] = jnp.asarray(informed_hazard_rate)
    specs["informed_shares_in_ages"] = jnp.asarray(informed_shares_in_ages)
    return specs


# Keep old function name for backward compatibility during transition
def add_informed_process_specs(specs, path_dict):
    """Legacy function name - redirects to new function."""
    return add_erp_belief_specs(specs, path_dict)
