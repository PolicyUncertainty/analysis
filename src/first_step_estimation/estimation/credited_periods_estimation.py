import numpy as np
import pandas as pd
import statsmodels.api as sm

from process_data.first_step_sample_scripts.create_credited_periods_est_sample import (
    create_credited_periods_est_sample,
)


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

    # # Create missing variables
    # df["const_men"] = 1 * (1-df["sex"])
    # df["const_women"] = 1 * df["sex"]
    df["experience_men"] = df["experience"] * (1 - df["sex"])
    df["experience_women"] = df["experience"] * df["sex"]
    # df["experience_sq_men"] = (df["experience"] ** 2) * (1-df["sex"])
    # df["experience_cub_men"] = (df["experience"] ** 3) * (1-df["sex"])
    #
    #
    # # Define columns for estimation
    # columns = [
    #     "const_men",
    #     "const_women",
    #     "experience_men",
    #     "experience_sq_men",
    #     "experience_cub_men",
    #     "experience_women",
    # ]
    #
    # # Fit OLS model
    # X = df[columns]
    # Y = df["credited_periods"]
    # model = sm.OLS(Y, X).fit()
    #
    # # Print model summary
    # print("Credited Periods Estimation Results:")
    # print("=" * 50)
    # print(model.summary())
    # def predict_men(exp):
    #     return model.params["const_men"] + model.params["experience_men"] * exp + model.params["experience_sq_men"] * exp ** 2 + model.params["experience_cub_men"] * exp ** 3

    df["very_long"] = df["credited_periods"] >= 45

    mean_exp_per_sex = (
        df.groupby(["sex", "very_long"])["experience"].mean().loc[(slice(None), True)]
    )

    params = pd.Series(
        {
            "experience_men": 45 / mean_exp_per_sex.loc[0],
            "experience_women": 45 / mean_exp_per_sex.loc[1],
        }
    )

    # Prepare estimates DataFrame
    # estimates = pd.DataFrame(params, columns=['estimate'])

    # Save estimates to CSV
    out_file_path = paths_dict["first_step_results"] + "credited_periods_estimates.csv"
    params.to_csv(out_file_path)

    # Save additional data for plotting if needed
    df_with_predictions = df.copy()
    df_with_predictions["predicted_credited_periods"] = (
        df_with_predictions["experience_men"] * params["experience_men"]
        + df_with_predictions["experience_women"] * params["experience_women"]
    )
    plot_data_path = paths_dict["first_step_results"] + "credited_periods_plot_data.csv"
    df_with_predictions.to_csv(plot_data_path, index=False)

    return params
