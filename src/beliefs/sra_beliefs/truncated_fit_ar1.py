"""
Plot to illustrate the fit of the AR1 process for SRA beliefs.

This plot shows:
- 5-year average expected values from the truncated normal distributions
- Standard deviations from the variance of the truncated normal
- AR1 predictions with dashed lines for standard deviations (similar to zeppelin graph)
- Horizontal bars for means and dotted lines for standard deviations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from beliefs.sra_beliefs.random_walk import filter_df
from set_styles import get_figsize, set_colors

JET_COLOR_MAP, LINE_STYLES = set_colors()


def plot_ar1_fit(
    path_dict,
    show=False,
    save=False,
):
    """
    Plot the fit of the AR1 process to the truncated normal distributions.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing file paths
    df : pd.DataFrame, optional
        DataFrame with truncated normal parameters (loaded if None)
    show : bool, default False
        Whether to display the plot
    save : bool, default False
        Whether to save the plot
    """
    df = pd.read_csv(path_dict["beliefs_data"] + "soep_is_truncated_normals.csv")

    # Filter data (same as in random_walk.py)
    df = filter_df(df)

    # Load estimated AR1 parameters
    params_df = pd.read_csv(path_dict["beliefs_est_results"] + "beliefs_parameters.csv")
    alpha = params_df[params_df["parameter"] == "alpha"]["estimate"].values[0]
    sigma_sq = params_df[params_df["parameter"] == "sigma_sq"]["estimate"].values[0]

    end_age_plot = 63
    start_age_plot = 25
    age_bins = np.arange(start_age_plot, end_age_plot + 1, 5)
    # Create age groups of 5 years
    df["age_group"] = pd.cut(
        df["age"],
        bins=age_bins,
        labels=[f"{i}-{i + 4}" for i in age_bins[:-1]],
        include_lowest=True,
    )

    # Calculate 5-year averages for expected value and standard deviation
    grouped = df.groupby("age_group", observed=True)
    mean_exp_val = grouped["ex_val"].mean()
    mean_std = grouped["var"].apply(lambda x: np.sqrt(x.mean()))

    # Get the midpoint age for each group for plotting
    age_midpoints = age_bins + 2.5

    # Generate AR1 predictions
    # Starting from age 30 to age 70
    ages = np.arange(start_age_plot, end_age_plot, 1)

    SRA_t = np.ones(ages.shape) * 67
    ar1_prediction = SRA_t + (end_age_plot - ages) * alpha
    exp_SRA_resolution = SRA_t + (end_age_plot - ages) * alpha
    ar1_upper = exp_SRA_resolution + 1.96 * np.sqrt(sigma_sq) * np.sqrt(
        end_age_plot - ages
    )
    ar1_lower = exp_SRA_resolution - 1.96 * np.sqrt(sigma_sq) * np.sqrt(
        end_age_plot - ages
    )

    age_30_mask = ages == 30
    expect_at_30 = {
        "lower_bound": ar1_lower[age_30_mask][0],
        "expectation": ar1_prediction[age_30_mask][0],
        "upper_bound": ar1_upper[age_30_mask][0],
    }
    pd.Series(data=expect_at_30).to_csv(
        path_dict["beliefs_est_results"] + "expect_at_30.csv"
    )

    # Create the plot
    fig, ax = plt.subplots()

    # Plot AR1 predictions
    ax.plot(
        ages,
        ar1_prediction,
        color=JET_COLOR_MAP[0],
        linewidth=3,
        label="Expected SRA (AR1)",
    )

    # Plot AR1 standard deviation bounds with dashed lines (like zeppelin graph)
    ax.plot(
        ages,
        ar1_upper,
        color=JET_COLOR_MAP[0],
        linestyle="--",
        linewidth=2,
        label="95% CI (AR1)",
    )
    ax.plot(ages, ar1_lower, color=JET_COLOR_MAP[0], linestyle="--", linewidth=2)

    # Plot horizontal bars for mean of the means of truncated normal
    for i, (age_mid, mean_val, std_val) in enumerate(
        zip(age_midpoints, mean_exp_val, mean_std)
    ):
        # # Horizontal bar for the mean
        # ax.hlines(
        #     y=mean_val,
        #     xmin=age_mid - 2,
        #     xmax=age_mid + 2,
        #     color=JET_COLOR_MAP[3],
        #     linewidth=3,
        #     label='Mean (Truncated Normal)' if i == 0 else ''
        # )

        # Make diamonds for the mean
        ax.plot(
            age_mid,
            mean_val,
            marker="D",
            color=JET_COLOR_MAP[3],
            markersize=8,
            label="Mean (Trunc. Normal)" if i == 0 else "",
        )

        # Dotted lines for standard deviation
        ax.plot(
            [age_mid, age_mid],
            [mean_val - 1.96 * std_val, mean_val + 1.96 * std_val],
            color=JET_COLOR_MAP[3],
            linestyle=":",
            linewidth=2,
            label="Std (Trunc. Normal)" if i == 0 else "",
        )

    # Customize plot
    ax.set_xlabel("Age")
    ax.set_ylabel("Expected SRA")
    ax.set_ylim([65, 72])
    ax.legend(loc="best")

    plt.tight_layout()

    if save:
        plt.savefig(
            path_dict["beliefs_plots"] + "ar1_fit_plot.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()
    else:
        plt.close()
