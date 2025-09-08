import pandas as pd
import numpy as np
import statsmodels.api as sm
from process_data.first_step_sample_scripts.create_credited_periods_est_sample import create_credited_periods_est_sample


def calibrate_credited_periods(paths_dict, load_data=False):
    """
    Calibrate credited periods model using OLS regression.
    
    Parameters:
    -----------
    paths_dict : dict
        Dictionary containing file paths
    load_data : bool, default False
        Whether to load existing data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the estimated parameters
    """
    # Load and prepare data
    df = create_credited_periods_est_sample(paths_dict, load_data=load_data)

    # Create missing variables
    df["const"] = 1
    df["experience_men"] = df["experience"] * (1-df["sex"])
    df["experience_women"] = df["experience"] * df["sex"]
    
    # Define columns for estimation
    columns = [
        "experience_men",
        "experience_women",
    ]
    
    # Fit OLS model
    X = df[columns]
    Y = df["credited_periods"]
    model = sm.OLS(Y, X).fit()
    
    # Print model summary
    print("Credited Periods Estimation Results:")
    print("=" * 50)
    print(model.summary())
    
    # Prepare estimates DataFrame
    estimates = pd.DataFrame(model.params, columns=['estimate'])
    
    # Save estimates to CSV
    out_file_path = paths_dict["est_results"] + "credited_periods_estimates.csv"
    estimates.to_csv(out_file_path)
    
    # Save additional data for plotting if needed
    df_with_predictions = df.copy()
    df_with_predictions["predicted_credited_periods"] = model.predict(X)
    plot_data_path = paths_dict["est_results"] + "credited_periods_plot_data.csv"
    df_with_predictions.to_csv(plot_data_path, index=False)
    
    return estimates