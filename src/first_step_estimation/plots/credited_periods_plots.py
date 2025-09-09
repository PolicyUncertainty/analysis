import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load the data and results needed for plotting
from process_data.first_step_sample_scripts.create_credited_periods_est_sample import (
    create_credited_periods_est_sample,
)
from set_styles import set_colors


def plot_credited_periods_results(path_dict, specs, show=False, save=False):
    """
    Plot credited periods estimation results.

    Parameters:
    -----------
    path_dict : dict
        Dictionary containing file paths
    specs : dict
        Specifications for the plotting
    show : bool, default False
        Whether to display the plots
    save : bool, default False
        Whether to save the plots
    """

    # Recreate the estimation to get the data for plotting
    df = create_credited_periods_est_sample(path_dict, load_data=True)

    # Create variables as in the estimation
    df["const"] = 1
    df["experience_men"] = df["experience"] * (1 - df["sex"])
    df["experience_women"] = df["experience"] * df["sex"]

    columns = [
        "experience_men",
        "experience_women",
    ]

    # Fit the model to get predictions
    X = df[columns]
    Y = df["credited_periods"]
    model = sm.OLS(Y, X).fit()

    # Create the plot
    plot_credited_periods_vs_exp(
        df, model, columns, show=show, save=save, path_dict=path_dict
    )


def plot_credited_periods_vs_exp(
    df, model, columns, show=False, save=False, path_dict=None
):
    """
    Plot credited periods (actual + predicted) vs experience.

    Parameters:
    -----------
    df : pd.DataFrame
        Data containing credited periods and experience
    model : statsmodels regression model
        Fitted model for predictions
    columns : list
        Column names used in the model
    show : bool, default False
        Whether to display the plot
    save : bool, default False
        Whether to save the plot
    path_dict : dict, optional
        Dictionary containing file paths for saving
    """
    jet_colors, line_styles = set_colors()

    df["predicted_credited_periods"] = model.predict(df[columns])
    men_mask = df["sex"] == 0

    fig, ax = plt.subplots()

    ax.scatter(
        df[men_mask]["experience"],
        df[men_mask]["credited_periods"],
        label="Actual (Men)",
        color=jet_colors[0],
        alpha=0.6,
    )
    ax.scatter(
        df[~men_mask]["experience"],
        df[~men_mask]["credited_periods"],
        label="Actual (Women)",
        color=jet_colors[3],
        alpha=0.6,
    )
    ax.scatter(
        df[men_mask]["experience"],
        df[men_mask]["predicted_credited_periods"],
        label="Predicted (Men)",
        color=jet_colors[0],
        marker="^",
        s=30,
    )
    ax.scatter(
        df[~men_mask]["experience"],
        df[~men_mask]["predicted_credited_periods"],
        label="Predicted (Women)",
        color=jet_colors[3],
        marker="^",
        s=30,
    )

    ax.set_xlabel("Experience")
    ax.set_ylabel("Credited Periods")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save and path_dict:
        save_path = path_dict["first_step_plots"] + "credited_periods_vs_experience.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
