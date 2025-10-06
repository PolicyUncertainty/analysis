from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from set_styles import set_colors

JET_COLOR_MAP, LINE_STYLES = set_colors()


def calibrate_uninformed_hazard_rate_with_se(
    df, specs, calculate_se=False, n_bootstrap=1000, random_state=42
):
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
    results = []

    # Calibrate hazard rate for each education group
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        df_restricted = df[df["education"] == edu_val]

        observed_informed_shares, weights = generate_observed_informed_shares_age_bins(
            df_restricted
        )

        params = fit_moments(moments=observed_informed_shares, weights=weights)

        # Calculate bootstrapped standard errors if requested
        bootstrap_se = bootstrap_standard_errors(
            df_restricted,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            calculate_se=calculate_se,
        )
        results_edu = pd.DataFrame(
            {
                "parameter": ["initial_informed_share", "hazard_rate"],
                "type": [edu_label, edu_label],
                "estimate": params.values,
                "std_error": bootstrap_se,
            }
        )

        results.append(results_edu)

    results = pd.concat(results, ignore_index=True)
    return results


def filter_dataset(df, specs):
    """Restrict dataset to relevant age range, filter invalid beliefs and missing education"""
    relevant_cols = [
        "age",
        "informed",
        "fweights",
        "education",
    ]
    df = df[relevant_cols].copy()
    df["age"] = df["age"].astype(int)
    # missing education
    df = df[df["education"].notna()]
    # filter out individuals that are too old
    df = df[df["age"] <= specs["max_ret_age"]]
    return df


def generate_observed_informed_shares_age_bins(df):
    # Create five-year age bins
    df = df.copy()
    df["age_bin"] = (df["age"] // 5) * 5

    sum_fweights = df.groupby("age_bin")["fweights"].sum()
    informed_sum_fweights = pd.Series(index=sum_fweights.index, data=0, dtype=float)
    informed_sum_fweights.update(
        df[df["informed"] == 1].groupby("age_bin")["fweights"].sum()
    )
    informed_by_age = informed_sum_fweights / sum_fweights
    weights = sum_fweights / sum_fweights.sum()
    return informed_by_age, weights


def fit_moments(moments, weights):
    params_guess = np.array([0.1, 0.01])
    partial_obj = partial(objective_function_age_bins, moments=moments, weights=weights)
    result = minimize(fun=partial_obj, x0=params_guess, method="BFGS")
    params = pd.Series(index=["initial_informed_share", "hazard_rate"], data=result.x)
    return params


def objective_function_age_bins(params, moments, weights):
    observed_age_bins = moments.index.values
    predicted_shares = predicted_shares_by_age_bins(
        params=params, age_bins_to_predict=observed_age_bins
    )
    return (((predicted_shares - moments) ** 2) * weights).sum()


def predicted_shares_by_age_bins(params, age_bins_to_predict):
    # Get individual ages spanning all bins
    min_age = age_bins_to_predict.min()
    max_age = age_bins_to_predict.max() + 4  # +4 to cover the full last bin
    age_span = np.arange(min_age, max_age + 1)

    # Use constant hazard rate
    hazard_rate = params[1]
    predicted_hazard_rate = hazard_rate * np.ones_like(age_span, dtype=float)

    # Calculate informed shares at individual age level
    informed_shares = np.zeros_like(age_span, dtype=float)
    initial_informed_share = params[0]
    informed_shares[0] = initial_informed_share
    uninformed_shares = 1 - informed_shares

    for period in range(1, len(age_span)):
        uninformed_shares[period] = uninformed_shares[period - 1] * (
            1 - predicted_hazard_rate[period - 1]
        )
        informed_shares[period] = 1 - uninformed_shares[period]

    # Aggregate to bins by taking the mean of ages within each bin
    age_df = pd.DataFrame({"age": age_span, "informed_share": informed_shares})
    age_df["age_bin"] = (age_df["age"] // 5) * 5
    binned_shares = age_df.groupby("age_bin")["informed_share"].mean()

    # Return only the requested bins
    relevant_shares = binned_shares.loc[age_bins_to_predict]
    return relevant_shares


def bootstrap_standard_errors(
    df_restricted, n_bootstrap=1000, random_state=42, calculate_se=True
):
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
    if not calculate_se:
        return [pd.NA, pd.NA]

    np.random.seed(random_state)

    # Store parameter estimates from each bootstrap sample
    bootstrap_estimates = []

    fail_count = 0
    for i in range(n_bootstrap):
        try:
            # Create bootstrap sample by sampling with replacement
            bootstrap_sample = df_restricted.sample(
                n=len(df_restricted),
                replace=True,
                weights=df_restricted["fweights"],
                random_state=random_state + i,  # Different seed for each iteration
            ).copy()

            # Generate observed informed shares for this bootstrap sample
            observed_informed_shares, weights = (
                generate_observed_informed_shares_age_bins(bootstrap_sample)
            )

            # Fit moments for this bootstrap sample
            params = fit_moments(moments=observed_informed_shares, weights=weights)

            # Store the parameter estimates
            bootstrap_estimates.append(params.values)
        except:
            fail_count += 1
            continue
    print(f"  Successful bootstrap samples: {n_bootstrap - fail_count}/{n_bootstrap}")
    std_errors = np.std(bootstrap_estimates, axis=0, ddof=1)
    return std_errors
