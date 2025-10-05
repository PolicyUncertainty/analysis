import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from beliefs.soep_is.belief_data_plots import create_gebjahr_groups
from set_styles import set_colors


def plot_predicted_vs_actual_means(
    path_dict, specs, show=False, save=False, df=None, params=None, censor_above=None
):
    JET_COLOR_MAP, LINE_STYLES = set_colors()
    """Plot the predicted vs actual mean ERP beliefs by age in 2022."""
    # Load data
    if df is None:
        df = pd.read_csv(
            path_dict["beliefs_data"] + "soep_is_clean.csv", dtype={"gebjahr": int}
        )
    if params is None:
        params = pd.read_csv(
            path_dict["beliefs_est_results"] + "beliefs_parameters.csv"
        )

    # Filter relevant data for actual means (same as violin plot)
    relevant_columns = [
        "belief_pens_deduct",
        "belief_pens_deduct_rob_times1_5",
        "belief_pens_deduct_rob_times0_5",
        "gebjahr",
    ]
    data_deduction = df[~df["belief_pens_deduct"].isnull()][relevant_columns]

    # Apply censoring if specified (same as violin plot)
    if censor_above is not None:
        data_deduction = data_deduction.copy()
        data_deduction["belief_pens_deduct"] = data_deduction[
            "belief_pens_deduct"
        ].clip(upper=censor_above)

    # Create age bins (same as violin plot)
    age_bins = [-np.inf] + list(range(1957, 2001, 5))
    data_deduction["gebjahr_group"] = create_gebjahr_groups(
        data_deduction, age_bins=age_bins
    )

    # Calculate actual means and medians by cohort group (same as violin plot)
    ded_data_grouped = data_deduction.groupby(["gebjahr_group"], observed=True)
    ded_data_mean = ded_data_grouped["belief_pens_deduct"].mean()
    ded_data_median = ded_data_grouped["belief_pens_deduct"].median()

    # Generate predicted means
    initial_age = df["age"].min()
    ages_to_predict = np.arange(initial_age, specs["max_ret_age"] + 1)

    # Initialize DataFrame to hold predicted means
    predicted_means = pd.DataFrame(columns=specs["education_labels"])

    for edu_val, edu_label in enumerate(specs["education_labels"]):
        # Filter data for the current education level
        params_edu = params[params["type"] == edu_label]

        # Generate predicted shares
        predicted_shares_edu = predicted_shares_by_age(
            params=params_edu, ages_to_predict=ages_to_predict
        )

        # Calculate predicted means
        erp_uninformed = params_edu[params_edu["parameter"] == "erp_uninformed_belief"][
            "estimate"
        ].values[0]
        predicted_means[edu_label] = (
            predicted_shares_edu * specs["ERP"]
            + (1 - predicted_shares_edu) * erp_uninformed
        )

    # Create the plot
    fig, ax = plt.subplots()

    # Define age ranges in 2022 corresponding to birth cohorts
    # Birth cohorts: [<1957, 1957-1961, 1962-1966, 1967-1971, 1972-1976, 1977-1981, 1982-1986, 1987-1991, 1992-1996]
    # Ages in 2022: [>65, 61-65, 56-60, 51-55, 46-50, 41-45, 36-40, 31-35, 26-30]
    age_ranges = [
        (65, 70),  # 1956 & before -> 66+ in 2022
        (61, 65),  # 1957-1961 -> 61-65 in 2022
        (56, 60),  # 1962-1966 -> 56-60 in 2022
        (51, 55),  # 1967-1971 -> 51-55 in 2022
        (46, 50),  # 1972-1976 -> 46-50 in 2022
        (41, 45),  # 1977-1981 -> 41-45 in 2022
        (36, 40),  # 1982-1986 -> 36-40 in 2022
        (31, 35),  # 1987-1991 -> 31-35 in 2022
        (26, 30),  # 1992-1996 -> 26-30 in 2022
    ]

    # Plot actual means as horizontal lines (same format as violin plot)
    for i, (mean_val, median_val) in enumerate(
        zip(ded_data_mean.values, ded_data_median.values)
    ):
        if i < len(age_ranges):
            age_start, age_end = age_ranges[i]
            # Plot mean as horizontal line
            ax.hlines(
                y=mean_val,
                xmin=age_start,
                xmax=age_end,
                color=JET_COLOR_MAP[3],
                linewidth=3,
                label="mean" if i == 0 else "",
            )
            # Plot median as horizontal line
            ax.hlines(
                y=median_val,
                xmin=age_start,
                xmax=age_end,
                color=JET_COLOR_MAP[1],
                linewidth=3,
                linestyle="--",
                label="median" if i == 0 else "",
            )

    # Plot predicted means as dashed lines for each education level
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        ax.plot(
            ages_to_predict,
            predicted_means[edu_label],
            label=f"{edu_label} (predicted)",
            color=JET_COLOR_MAP[edu_val + 4],  # Offset to avoid color conflicts
            linestyle="--",
            linewidth=2,
        )

    # Add true ERP line
    ax.axhline(
        y=specs["ERP"], color="black", linestyle="--", linewidth=2, label="true ERP"
    )

    # Customize plot
    ax.set_xlabel("Age in 2022")
    ax.set_ylabel("Penalty in %")
    ax.set_ylim([0, 20])
    ax.set_yticks(np.arange(0, 21, 2.5))
    ax.set_xlim([25, 70])
    ax.legend(loc="upper left")

    fig.tight_layout()
    if save:
        plt.savefig(
            path_dict["beliefs_plots"] + "predicted_vs_actual_means_by_age.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()


def plot_predicted_vs_actual_informed_share(
    path_dict, specs, show=False, save=False, df=None, params=None
):
    JET_COLOR_MAP, LINE_STYLES = set_colors()
    """Plot the predicted vs actual informed shares by education level."""
    # Load data
    if df is None:
        df = pd.read_csv(path_dict["beliefs_data"] + "soep_is_clean.csv")
    if params is None:
        params = pd.read_csv(
            path_dict["beliefs_est_results"] + "beliefs_parameters.csv"
        )

    # Generate predicted and actual informed shares
    initial_age = df["age"].min()
    ages_to_predict = np.arange(initial_age, specs["max_ret_age"] + 1)

    # Initialize DataFrame to hold predicted shares
    observed_shares = pd.DataFrame(
        index=ages_to_predict, columns=specs["education_labels"]
    )
    predicted_shares = pd.DataFrame(columns=specs["education_labels"])
    # Store raw fweights sums for marker sizing
    fweights_dict = {}

    for edu_val, edu_label in enumerate(specs["education_labels"]):
        # Filter data for the current education level
        df_restricted = df[df["education"] == edu_val]
        params_edu = params[params["type"] == edu_label]

        # Generate observed shares, weights, and raw fweights
        observed_shares_edu, weights, sum_fweights = generate_observed_informed_shares(
            df_restricted
        )

        # Generate predicted shares
        predicted_shares_edu = predicted_shares_by_age(
            params=params_edu, ages_to_predict=ages_to_predict
        )

        # Update the DataFrames with the results for the current education level
        observed_shares[edu_label] = observed_shares_edu
        predicted_shares[edu_label] = predicted_shares_edu
        # Store raw fweights sums for this education level
        fweights_dict[edu_label] = sum_fweights

    # Create plot
    fig, ax = plt.subplots()

    # Calculate marker size scaling parameters using raw fweights
    all_fweights = pd.concat(fweights_dict.values())
    min_fweight = all_fweights.min()
    max_fweight = all_fweights.max()
    # Scale marker sizes
    min_marker_size = 5
    max_marker_size = 100

    for edu_val, edu_label in enumerate(specs["education_labels"]):
        # Get observed shares (no rolling mean)
        observed_shares_edu = observed_shares[edu_label]

        # Calculate marker sizes proportional to raw fweights
        fweights_edu = fweights_dict[edu_label]
        # Normalize fweights to marker size range
        if max_fweight > min_fweight:  # Avoid division by zero
            normalized_fweights = (fweights_edu - min_fweight) / (
                max_fweight - min_fweight
            )
            marker_sizes = min_marker_size + normalized_fweights * (
                max_marker_size - min_marker_size
            )
        else:
            marker_sizes = pd.Series(index=fweights_edu.index, data=min_marker_size)

        # Create scatter plot with variable marker sizes
        # Only plot points where we have valid observed data
        valid_idx = observed_shares_edu.notna()
        valid_ages = observed_shares_edu.index[valid_idx]

        ax.scatter(
            valid_ages,
            observed_shares_edu.loc[valid_ages],
            s=marker_sizes.reindex(valid_ages, fill_value=min_marker_size).values,
            label=f"Obs. {edu_label}",
            color=JET_COLOR_MAP[edu_val],
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Plot predicted line
        ax.plot(
            predicted_shares[edu_label],
            color=JET_COLOR_MAP[edu_val],
            label=f"Est. {edu_label}",
        )

    # Set labels
    ax.set_xlabel("Age")
    ax.set_ylabel("Share Informed")
    ax.legend()

    # Add a note about marker sizes
    ax.text(
        0.02,
        0.98,
        "Marker size ∝ Sample size",
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="top",
        alpha=0.7,
    )

    if save:
        plt.savefig(
            path_dict["beliefs_plots"] + "predicted_vs_actual_informed_share.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()


def generate_observed_informed_shares(df):
    """Generate observed informed shares by age from the DataFrame."""
    sum_fweights = df.groupby("age")["fweights"].sum()
    informed_sum_fweights = pd.Series(index=sum_fweights.index, data=0, dtype=float)
    informed_sum_fweights.update(
        df[df["informed"] == 1].groupby("age")["fweights"].sum()
    )
    informed_by_age = informed_sum_fweights / sum_fweights
    weights = sum_fweights / sum_fweights.sum()
    return informed_by_age, weights, sum_fweights


def predicted_shares_by_age(params, ages_to_predict):
    age_span = np.arange(ages_to_predict.min(), ages_to_predict.max() + 1)
    # This could be more complicated with age specific hazard rates

    hazard_rate = params[params["parameter"] == "hazard_rate"]["estimate"].values[0]
    predicted_hazard_rate = hazard_rate * np.ones_like(age_span, dtype=float)

    informed_shares = np.zeros_like(age_span, dtype=float)
    initial_informed_share = params[params["parameter"] == "initial_informed_share"][
        "estimate"
    ].values[0]
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
