"""
Plot predicted informed shares by education level with average observed shares.

This module plots:
- Predicted informed shares by age for each education level (continuous lines)
- Average observed shares for groups of five ages (horizontal lines with error bars)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from beliefs.erp_beliefs.erp_plots import predicted_shares_by_age
from set_styles import set_colors


def plot_predicted_informed_shares_by_education(
    path_dict, specs, show=False, save=False, df=None, params=None, by_education=False
):
    """
    Plot predicted informed shares by education with observed averages.

    Shows predicted informed shares as continuous lines for each education level,
    and average observed shares as horizontal lines with error bars for age groups.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing file paths
    specs : dict
        Specifications dictionary
    show : bool, default False
        Whether to display the plot
    save : bool, default False
        Whether to save the plot
    df : pd.DataFrame, optional
        SOEP-IS dataset (loaded if None)
    params : pd.DataFrame, optional
        Belief parameters (loaded if None)
    by_education : bool, default True
        If True, plot education-specific observed means and standard errors.
        If False, plot overall observed means and standard errors (pooled across education levels).
    """
    JET_COLOR_MAP, LINE_STYLES = set_colors()

    # Load data if not provided
    if df is None:
        df = pd.read_csv(path_dict["beliefs_data"] + "soep_is_clean.csv")
    if params is None:
        params = pd.read_csv(
            path_dict["beliefs_est_results"] + "beliefs_parameters.csv"
        )

    # Generate predicted shares
    initial_age = df["age"].min()
    ages_to_predict = np.arange(initial_age, specs["max_ret_age"] + 1)

    # Initialize DataFrame to hold predicted shares
    predicted_shares = pd.DataFrame(columns=specs["education_labels"])

    # Calculate predicted shares for each education level
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        params_edu = params[params["type"] == edu_label]

        # Generate predicted shares using the hazard rate model
        predicted_shares_edu = predicted_shares_by_age(
            params=params_edu, ages_to_predict=ages_to_predict
        )
        predicted_shares[edu_label] = predicted_shares_edu

    # Calculate observed average shares by age groups
    age_bins = list(range(25, 66, 5))  # 5-year age bins from 25 to 70

    if by_education:
        # Create storage for average shares and standard errors by education
        avg_shares_by_edu = {}
        sem_shares_by_edu = {}

        for edu_val, edu_label in enumerate(specs["education_labels"]):
            # Filter data for the current education level
            df_edu = df[df["education"] == edu_val].copy()
            df_edu = df_edu[df_edu["informed"].notna()]

            # Create age groups
            df_edu["age_group"] = pd.cut(
                df_edu["age"],
                bins=age_bins,
                labels=range(len(age_bins) - 1),
                right=False,
            )

            # Calculate average informed share and SEM by age group
            grouped = df_edu.groupby("age_group", observed=True)
            avg_shares = grouped["informed"].mean()
            sem_shares = grouped["informed"].sem()

            avg_shares_by_edu[edu_label] = avg_shares
            sem_shares_by_edu[edu_label] = sem_shares
    else:
        # Calculate overall observed shares (pooled across education levels)
        df_pooled = df[df["informed"].notna()].copy()

        # Create age groups
        df_pooled["age_group"] = pd.cut(
            df_pooled["age"],
            bins=age_bins,
            labels=range(len(age_bins) - 1),
            right=False,
        )

        # Calculate average informed share and SEM by age group (pooled)
        grouped = df_pooled.groupby("age_group", observed=True)
        avg_shares_pooled = grouped["informed"].mean()
        sem_shares_pooled = grouped["informed"].sem()

    # Create the plot
    fig, ax = plt.subplots()

    # Define age ranges for horizontal lines (groups of 5 years)
    age_ranges = [(age_bins[i], age_bins[i + 1]) for i in range(len(age_bins) - 1)]

    # Plot for each education level
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        color = JET_COLOR_MAP[edu_val]

        # format edu label to lower letters only
        edu_label_lower = edu_label.lower()

        # Plot predicted shares as continuous line
        ax.plot(
            ages_to_predict,
            predicted_shares[edu_label],
            color=color,
            linewidth=2,
            label=f"{edu_label_lower}",
        )

        if by_education:
            # Plot education-specific average observed shares as horizontal lines with error bars
            avg_shares = avg_shares_by_edu[edu_label]
            sem_shares = sem_shares_by_edu[edu_label]

            for i, (age_start, age_end) in enumerate(age_ranges):
                if i in avg_shares.index:
                    mean_val = avg_shares.loc[i]
                    sem_val = sem_shares.loc[i] if i in sem_shares.index else 0

                    # # Plot horizontal line for the mean
                    # ax.hlines(
                    #     y=mean_val,
                    #     xmin=age_start,
                    #     xmax=age_end,
                    #     color=color,
                    #     linewidth=2,
                    #     linestyle="-",
                    #     alpha=0.6,
                    # )

                    # Plot error bars at the midpoint of the age range
                    # Use diamond markers as midpoint
                    age_midpoint = (age_start + age_end) / 2
                    age_midpoint += 0.5 * edu_val  # slight offset for visibility
                    ax.errorbar(
                        x=age_midpoint,
                        y=mean_val,
                        yerr=sem_val,  # 95% confidence interval
                        fmt="D",
                        color=color,
                        ecolor=color,
                        capsize=4,
                        markersize=5,
                        alpha=0.8,
                    )

    if not by_education:
        # Plot overall observed shares (same for all education levels)
        for i, (age_start, age_end) in enumerate(age_ranges):
            if i in avg_shares_pooled.index:
                mean_val = avg_shares_pooled.loc[i]
                sem_val = (
                    sem_shares_pooled.loc[i] if i in sem_shares_pooled.index else 0
                )

                # # Plot horizontal line for the mean (in gray/black)
                # ax.hlines(
                #     y=mean_val,
                #     xmin=age_start,
                #     xmax=age_end,
                #     color="black",
                #     linewidth=2.5,
                #     linestyle="-",
                #     alpha=0.6,
                #     label="Overall observed" if i == 0 else "",
                # )

                # Plot error bars at the midpoint of the age range
                age_midpoint = (age_start + age_end) / 2
                ax.errorbar(
                    x=age_midpoint,
                    y=mean_val,
                    yerr=sem_val,  # 95% confidence interval
                    fmt="D",
                    color="black",
                    ecolor="gray",
                    capsize=4,
                    markersize=6,
                    alpha=0.8,
                )

    # Customize plot
    ax.set_xlabel("Age")
    ax.set_ylabel("Share Informed")
    ax.set_xlim([25, 65])
    ax.set_ylim([0, 0.6])
    ax.set_yticks(np.arange(0, 0.7, 0.1))

    # Create legend
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()

    # Determine filename based on the by_education flag
    if by_education:
        filename = "predicted_informed_shares_by_education.png"
    else:
        filename = "predicted_informed_shares_overall.png"

    if save:
        plt.savefig(path_dict["beliefs_plots"] + filename, bbox_inches="tight", dpi=100)
    if show:
        plt.show()
    else:
        plt.close()
