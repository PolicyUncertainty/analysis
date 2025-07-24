import pandas as pd
import numpy as np
from beliefs.belief_formalization.random_walk import est_expected_SRA    

def raw_belief_heterogeneity_by_covariate(df, cov_list, born_after=1964, born_before=3000):
    """Computes mean and standard error of beliefs by covariate.
    Covariates must be binary. Df must contain truncated normals parameters.
    Belief variables (generated here): "individual_alpha", "individual_sigma_sq", informed."""
    df = construct_individual_sra_belief_params(df)
    df = filter_df(df, born_after=born_after, born_before=born_before)
    
    # Initialize results dictionary with statistics as keys
    results_dict = {"covariate": ["mean_alpha", "std_err_alpha", "mean_sigma_sq", 
                                  "std_err_sigma_sq", "mean_informed", "std_err_informed"]}
    
    for cov in cov_list:
        if cov not in df.columns:
            print(f"Warning: Covariate '{cov}' not found in DataFrame, skipping...")
            continue
        
        # Calculate statistics for each binary value (0 and 1)
        for val in [0, 1]:
            col_name = f"{cov}_{val}"
            subset = df[df[cov] == val]
            
            if len(subset) == 0:
                # If no observations for this value, fill with NaN
                results_dict[col_name] = [np.nan] * 6
                continue
            
            # Calculate mean and std error for individual_alpha
            mean_alpha = subset["individual_alpha"].mean()
            std_err_alpha = subset["individual_alpha"].std() / np.sqrt(len(subset))
            
            # Calculate mean and std error for individual_sigma_sq
            mean_sigma_sq = subset["individual_sigma_sq"].mean()
            std_err_sigma_sq = subset["individual_sigma_sq"].std() / np.sqrt(len(subset))
            
            # Calculate mean and std error for informed
            mean_informed = subset["informed"].mean()
            std_err_informed = subset["informed"].std() / np.sqrt(len(subset))
            
            # Add to results dictionary
            results_dict[col_name] = [mean_alpha, std_err_alpha, mean_sigma_sq, 
                                     std_err_sigma_sq, mean_informed, std_err_informed]
    
    # Convert to DataFrame
    results = pd.DataFrame(results_dict)
    return results


def construct_individual_sra_belief_params(df):
    """
    Constructs individual SRA belief parameters based on random walk assumption by simply dividing the expected SRA increase and forecast variance by the time to retirement.
    
    Args:
        df: DataFrame with truncated normals data.
    
    Returns:
        DataFrame with individual SRA belief parameters
    """

    df["exp_SRA_increase"] = df["ex_val"] - df["current_SRA"]
    df["individual_alpha"] = df["exp_SRA_increase"] / df["time_to_ret"]
    df["individual_sigma_sq"] = df["var"] / df["time_to_ret"]
    df["individual_sigma"] = np.sqrt(df["individual_sigma_sq"])
    return df

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
            paths["intermediate_data"] + "beliefs/soep_is_truncated_normals.csv"
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

def filter_df(df, born_after = 1964, born_before = 3000):
    """Drop observations of people born before 1964 and drop missing subjective expectation parameters."""
    df = df[df["gebjahr"] >= born_after]
    df = df[df["gebjahr"] < born_before]
    df = df.dropna(subset=["ex_val", "var", "fweights", "time_to_ret"])
    print(
        f"Filtered data: {len(df)} observations remaining after dropping birth years before 1964, and people with missing values in subjective expectation parameters."
    )
    return df


