from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from set_styles import set_colors
JET_COLOR_MAP, LINE_STYLES = set_colors()

def calibrate_uninformed_hazard_rate_with_se(df, specs, calculate_se=False, n_bootstrap=1000, random_state=42):
    """
    This functions calibrates the hazard rate of becoming informed for the uninformed
    individuals with method of (weighted) moments.

    Informed (ERP<=5) are classified in "process_soep_is".
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    specs : dict
        Specifications dictionary
    n_bootstrap : int
        Number of bootstrap samples (default: 1000)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    pd.DataFrame : Results with parameter estimates and bootstrapped standard errors
    """
    
    df = filter_dataset(df, specs)

    # Create empty df 
    results = pd.DataFrame(columns=["parameter", "type", "estimate", "std_error"])

    # Calibrate hazard rate for each education group
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        df_restricted = df[df["education"] == edu_val]

        # Estimate the average ERP belief of the uninformed individuals
        avg_belief_uninformed, sem_uninformed = beliefs_of_uninformed(df_restricted)

        # Estimate the initial share and hazard rate (original estimates)
        observed_informed_shares, weights = generate_observed_informed_shares(df_restricted)
        params = fit_moments(moments=observed_informed_shares, weights=weights)
        
        # Calculate bootstrapped standard errors if requested
        if calculate_se:
            print(f"Calculating bootstrap standard errors for {edu_label}...")
            bootstrap_se = bootstrap_standard_errors(
                df_restricted, 
                n_bootstrap=n_bootstrap, 
                random_state=random_state
            )
            print(f"  Successful bootstrap samples: {bootstrap_se['n_successful_bootstraps']}/{n_bootstrap}")
            
            std_errors = [
                bootstrap_se['initial_informed_share'], 
                bootstrap_se['hazard_rate']
            ]
        else:
            std_errors = [pd.NA, pd.NA]

        # Append results for each parameter
        results = pd.concat([
            results,
            pd.DataFrame({
                "parameter": ["initial_informed_share", "hazard_rate"],
                "type": [edu_label, edu_label],
                "estimate": params.values,
                "std_error": std_errors,
            })
        ], ignore_index=True)

    return results


def filter_dataset(df, specs):
    """Restrict dataset to relevant age range, filter invalid beliefs and missing education"""
    relevant_cols = [
        "belief_pens_deduct",
        "age",
        "informed",
        "fweights",
        "education",
    ]
    df = df[relevant_cols].copy()
    df["age"] = df["age"].astype(int)
    # invalid beliefs 
    df = df[df["belief_pens_deduct"] >= 0]
    # missing education
    df = df[df["education"].notna()]
    # filter out individuals that are too old
    df = df[df["age"] <= specs["max_ret_age"]]
    return df


def beliefs_of_uninformed(df):
    """This function saves the average ERP belief of the uninformed individuals in the
    dataset."""
    df_u = df[df["informed"] == 0]
    normalized_fweights = (df_u["fweights"] / df_u["fweights"].sum()) * df_u.shape[0]
    weighted_beliefs = df_u["belief_pens_deduct"] * normalized_fweights
    weighted_belief_uninformed = weighted_beliefs.mean()
    sem = weighted_beliefs.sem()

    return weighted_belief_uninformed, sem


def generate_observed_informed_shares(df):
    sum_fweights = df.groupby("age")["fweights"].sum()
    informed_sum_fweights = pd.Series(index=sum_fweights.index, data=0, dtype=float)
    informed_sum_fweights.update(
        df[df["informed"] == 1].groupby("age")["fweights"].sum()
    )
    informed_by_age = informed_sum_fweights / sum_fweights
    weights = sum_fweights / sum_fweights.sum()
    return informed_by_age, weights


def fit_moments(moments, weights):
    params_guess = np.array([0.1, 0.01])
    partial_obj = partial(objective_function, moments=moments, weights=weights)
    result = minimize(fun=partial_obj, x0=params_guess, method="BFGS")
    params = pd.Series(index=["initial_informed_share", "hazard_rate"], data=result.x)
    return params


def objective_function(params, moments, weights):
    observed_ages = moments.index.values
    predicted_shares = predicted_shares_by_age(
        params=params, ages_to_predict=observed_ages
    )
    return (((predicted_shares - moments) ** 2) * weights).sum()


def predicted_shares_by_age(params, ages_to_predict):
    age_span = np.arange(ages_to_predict.min(), ages_to_predict.max() + 1)
    # The next line could be more complicated with age specific hazard rates
    # For now we use constant
    hazard_rate = params[1]
    predicted_hazard_rate = hazard_rate * np.ones_like(age_span, dtype=float)

    informed_shares = np.zeros_like(age_span, dtype=float)
    initial_informed_share = params[0]
    informed_shares[0] = initial_informed_share
    uninformed_shares = 1 - informed_shares

    for period in range(1, len(age_span)):
        uninformed_shares[period] = uninformed_shares[period - 1] * (
            1 - predicted_hazard_rate[period - 1]
        )
        informed_shares[period] = 1 - uninformed_shares[period]

    relevant_shares = pd.Series(index=age_span, data=informed_shares).loc[
        ages_to_predict
    ]
    return relevant_shares


def bootstrap_standard_errors(df_restricted, n_bootstrap=1000, random_state=42):
    """
    Estimate bootstrapped standard errors for the hazard rate parameters.
    
    Parameters:
    -----------
    df_restricted : pd.DataFrame
        Filtered dataset for a specific education group
    n_bootstrap : int
        Number of bootstrap samples (default: 1000)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict : Dictionary with standard errors for 'initial_informed_share' and 'hazard_rate'
    """
    np.random.seed(random_state)
    
    # Store parameter estimates from each bootstrap sample
    bootstrap_estimates = []
    
    for i in range(n_bootstrap):
        try:
            # Create bootstrap sample by sampling with replacement
            bootstrap_sample = df_restricted.sample(
                n=len(df_restricted), 
                replace=True, 
                weights=df_restricted['fweights'],
                random_state=random_state + i  # Different seed for each iteration
            ).copy()
            
            # Generate observed informed shares for this bootstrap sample
            observed_informed_shares, weights = generate_observed_informed_shares(bootstrap_sample)
            
            # Skip if we don't have enough variation in the bootstrap sample
            if len(observed_informed_shares) < 2 or observed_informed_shares.isna().all():
                continue
                
            # Fit moments for this bootstrap sample
            params = fit_moments(moments=observed_informed_shares, weights=weights)
            
            # Store the parameter estimates
            bootstrap_estimates.append(params.values)
            
        except Exception as e:
            # Skip failed bootstrap samples (convergence issues, etc.)
            continue
    
    # Convert to numpy array for easier calculation
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    # Calculate standard errors as standard deviation across bootstrap samples
    if len(bootstrap_estimates) > 0:
        std_errors = np.std(bootstrap_estimates, axis=0, ddof=1)
        return {
            'initial_informed_share': std_errors[0],
            'hazard_rate': std_errors[1],
            'n_successful_bootstraps': len(bootstrap_estimates)
        }
    else:
        # Return NaN if no successful bootstrap samples
        return {
            'initial_informed_share': np.nan,
            'hazard_rate': np.nan,
            'n_successful_bootstraps': 0
        }


