import numpy as np
import pandas as pd
from statsmodels import api as sm


def est_SRA_params(paths, df=None, print_summary=False):
    # load (if df is None) and filter data
    if df is None:
        df = pd.read_csv(
            paths["beliefs_data"] + "soep_is_truncated_normals.csv"
        )
    df = filter_df(df)
    # estimate expected SRA increase and variance
    alpha_hat, alpha_hat_std_err = est_expected_SRA(paths, df, print_summary=print_summary)
    sigma_sq_hat, sigma_sq_hat_std_err = estimate_expected_SRA_variance(paths, df)
    columns = ["parameter", "estimate", "std_error"]
    results_df = pd.DataFrame(
        columns=columns,
        data=[
            ["alpha", alpha_hat[0], alpha_hat_std_err[0]],
            ["sigma_sq", sigma_sq_hat[0], sigma_sq_hat_std_err[0]],
        ],
    )
    return results_df


def est_expected_SRA(paths, df=None, print_summary=False, covariates=None, include_constant=False):
    """
    Estimate expected SRA increase regression.
    
    Args:
        paths: Path dictionary
        df: DataFrame with data (should already be cleaned)
        print_summary: Whether to print regression summary
        covariates: List of covariate column names to include in regression
        include_constant: Whether to include constant term
    
    Returns:
        Tuple of (coefficients, standard_errors)
    """
    # set up regression
    base_x_var = "time_to_ret"
    weights = "fweights"
    
    # Create exp_SRA_increase if it doesn't exist
    if "exp_SRA_increase" not in df.columns:
        df["exp_SRA_increase"] = df["ex_val"] - df["current_SRA"]
    
    y_var = "exp_SRA_increase"
    
    # Build design matrix
    X_vars = [base_x_var]
    if covariates is not None:
        X_vars.extend(covariates)
    
    # Use the cleaned dataframe 
    X = df[X_vars].values
    y = df[y_var].values
    w = df[weights].values
    
    # Add constant if requested
    if include_constant:
        X = sm.add_constant(X)
        var_names = ['const'] + X_vars
    else:
        var_names = X_vars
    
    # Final check for any remaining inf/nan values
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(w)
    X = X[valid_mask]
    y = y[valid_mask]
    w = w[valid_mask]
    
    if len(X) == 0:
        raise ValueError("No valid observations remaining after cleaning")
    
    # regress expected SRA increase
    model = sm.WLS(exog=X, endog=y, weights=w)
    
    fitted_model = model.fit()
    
    if print_summary:

        # summarize distributions of dependent and independent variables
        print("\nDescriptive statistics:")
        desc_stats = df[[y_var] + X_vars].describe()
        print(desc_stats)

        # print model summary
        print(fitted_model.summary())
        coef_str = " + ".join([f"{coef:.4f} * {var}" for coef, var in zip(fitted_model.params, var_names)])
        print(f"Estimated regression equation: E[ret age change] = {coef_str}")
        print(f"Number of observations: {len(X)}")
    
    return fitted_model.params, fitted_model.bse


def est_alpha_heterogeneity(paths, df=None):
    """
    Estimate heterogeneity in alpha (expected SRA increase) by demographic covariates.
    
    Args:
        paths: Path dictionary
        df: DataFrame with data
        
    Returns:
        DataFrame with estimation results for all covariates and specifications
    """
    # load (if df is None) and filter data
    if df is None:
        df = pd.read_csv(
            paths["beliefs_data"] + "soep_is_truncated_normals.csv"
        )
    df = filter_df(df)
    
    # Define covariates to analyze
    covariates = ["sex", "education", "health"]
    
    results_list = []
    
    for covariate in covariates:
        # Check if covariate exists in data
        if covariate not in df.columns:
            print(f"Warning: {covariate} not found in data, skipping...")
            continue
        
        print(f"\nProcessing covariate: {covariate}")
        
        # Create a clean dataset for this covariate
        df_clean = df.copy()
        
        # Create exp_SRA_increase if not exists
        df_clean["exp_SRA_increase"] = df_clean["ex_val"] - df_clean["current_SRA"]
        
        # Add age if needed
        if 'age' not in df_clean.columns and 'gebjahr' in df_clean.columns:
            current_year = 2020  # Adjust as needed
            df_clean['age'] = current_year - df_clean['gebjahr']
        
        # Univariate regression (time_to_ret + covariate only)
        try:
            # Select required columns and drop rows with inf/nan
            required_cols_univ = ["time_to_ret", covariate, "exp_SRA_increase", "fweights"]
            df_reg_univ = df_clean[required_cols_univ].copy()
            
            # Replace inf with nan, then drop all rows with nan/inf
            df_reg_univ = df_reg_univ.replace([np.inf, -np.inf], np.nan)
            df_reg_univ = df_reg_univ.dropna()
            
            print(f"  Univariate regression: {len(df_reg_univ)} observations after cleaning")
            
            if len(df_reg_univ) > 0:
                params_univ, se_univ = est_expected_SRA(
                    paths, df_reg_univ, covariates=[covariate], include_constant=False
                )
                
                # Extract covariate coefficient (second coefficient after time_to_ret)
                covariate_coef_univ = params_univ[1] if len(params_univ) > 1 else np.nan
                covariate_se_univ = se_univ[1] if len(se_univ) > 1 else np.nan
                
                results_list.append({
                    "covariate": covariate,
                    "specification": "univariate",
                    "coefficient": covariate_coef_univ,
                    "std_error": covariate_se_univ,
                    "t_stat": covariate_coef_univ / covariate_se_univ if covariate_se_univ != 0 else np.nan,
                    "ci_lower": covariate_coef_univ - 1.96 * covariate_se_univ,
                    "ci_upper": covariate_coef_univ + 1.96 * covariate_se_univ,
                    "n_obs": len(df_reg_univ)
                })
            else:
                print(f"  No valid observations for univariate regression with {covariate}")
                
        except Exception as e:
            print(f"Error in univariate regression for {covariate}: {e}")
        
        # Regression controlling for age (time_to_ret + covariate + age controls)
        try:
            # Select required columns including age control
            age_controls = ['age'] if 'age' in df_clean.columns else []
            
            if age_controls:
                required_cols_ctrl = ["time_to_ret", covariate] + age_controls + ["exp_SRA_increase", "fweights"]
                df_reg_ctrl = df_clean[required_cols_ctrl].copy()
                
                # Replace inf with nan, then drop all rows with nan/inf
                df_reg_ctrl = df_reg_ctrl.replace([np.inf, -np.inf], np.nan)
                df_reg_ctrl = df_reg_ctrl.dropna()
                
                print(f"  Age-controlled regression: {len(df_reg_ctrl)} observations after cleaning")
                
                if len(df_reg_ctrl) > 0:
                    params_ctrl, se_ctrl = est_expected_SRA(
                        paths, df_reg_ctrl, covariates=[covariate] + age_controls, include_constant=False
                    )
                    
                    # Extract covariate coefficient (second coefficient after time_to_ret)
                    covariate_coef_ctrl = params_ctrl[1] if len(params_ctrl) > 1 else np.nan
                    covariate_se_ctrl = se_ctrl[1] if len(se_ctrl) > 1 else np.nan
                    
                    results_list.append({
                        "covariate": covariate,
                        "specification": "with_age_control",
                        "coefficient": covariate_coef_ctrl,
                        "std_error": covariate_se_ctrl,
                        "t_stat": covariate_coef_ctrl / covariate_se_ctrl if covariate_se_ctrl != 0 else np.nan,
                        "ci_lower": covariate_coef_ctrl - 1.96 * covariate_se_ctrl,
                        "ci_upper": covariate_coef_ctrl + 1.96 * covariate_se_ctrl,
                        "n_obs": len(df_reg_ctrl)
                    })
                else:
                    print(f"  No valid observations for age-controlled regression with {covariate}")
            else:
                print(f"  Age variable not available for {covariate} age-controlled regression")
                
        except Exception as e:
            print(f"Error in age-controlled regression for {covariate}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    return results_df


def estimate_expected_SRA_variance(paths, df=None, print_summary=False):
    # set up regression
    x_var = "time_to_ret"
    weights = "fweights"
    y_var = "var"
    # regress estimated variance on time to retirement without constant
    model = sm.WLS(
        exog=df[x_var].values,
        endog=df[y_var].values,
        weights=df[weights].values,
    )
    if print_summary:
        print(model.fit().summary())
        print(
            f"Estimated regression equation: E[ret age variance] = "
            f"{sigma_sq_hat[0]} * (Time to retirement)"
        )
    sigma_sq_hat = model.fit().params
    sigma_sq_hat_std_err = model.fit().bse
    return sigma_sq_hat, sigma_sq_hat_std_err


def estimate_expected_SRA_variance_by_taking_average(paths, df=None):
    if df is None:
        df = pd.read_csv(
            paths["beliefs_data"] + "soep_is_truncated_normals.csv"
        )

    # truncate data: remove birth years outside before 1964
    df = filter_df(df)

    # divide estimated variances by time to retirement
    sigma_sq_hat = df["var"] / df["time_to_ret"]

    # weight and take average
    sigma_sq_hat = (sigma_sq_hat * df["fweights"]).sum() / df["fweights"].sum()
    sigma_sq_hat = np.array([sigma_sq_hat])

    sigma_sq_hat_std_err = np.std(df["var"] / df["time_to_ret"]) / np.sqrt(len(df))

    return sigma_sq_hat, sigma_sq_hat_std_err


def filter_df(df):
    """Drop observations of people born before 1964 and drop missing subjective expectation parameters."""
    df = df[df["gebjahr"] >= 1964]
    df = df.dropna(subset=["ex_val", "var", "fweights", "time_to_ret"])
    print(
        f"Filtered data: {len(df)} observations remaining after dropping birth years before 1964, and people with missing values in subjective expectation parameters."
    )
    return df