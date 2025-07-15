import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from export_results.figures.color_map import JET_COLOR_MAP

def plot_predicted_vs_actual_informed_share(path_dict, specs, show = False, df= None, params = None):
    """Plot the predicted vs actual informed shares by education level."""
    # Load data
    if df is None:
        df = pd.read_csv(path_dict["beliefs_data"] + "soep_is_clean.csv")
    if params is None:
        params = pd.read_csv(path_dict["beliefs_data"] + "beliefs_parameters.csv")
    
    # Generate predicted and actual informed shares
    initial_age = df["age"].min()
    ages_to_predict = np.arange(initial_age, specs["max_ret_age"] + 1)
    
    # Initialize DataFrame to hold predicted shares
    observed_shares = pd.DataFrame(index=ages_to_predict, columns=specs["education_labels"])
    predicted_shares = pd.DataFrame(columns=specs["education_labels"])
    # Store raw fweights sums for marker sizing
    fweights_dict = {}

    for edu_val, edu_label in enumerate(specs["education_labels"]):
        # Filter data for the current education level
        df_restricted = df[df["education"] == edu_val]
        params_edu = params[params["type"] == edu_label]
        
        # Generate observed shares, weights, and raw fweights
        observed_shares_edu, weights, sum_fweights = generate_observed_informed_shares(df_restricted)
        
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
    plt.rcParams.update(
        {
            "axes.titlesize": 30,
            "axes.labelsize": 30,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 30,
        }
    )
    # Make lines of plots thicker
    plt.rcParams["lines.linewidth"] = 3
    fig, ax = plt.subplots(figsize=(16, 9))
    
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
            normalized_fweights = (fweights_edu - min_fweight) / (max_fweight - min_fweight)
            marker_sizes = min_marker_size + normalized_fweights * (max_marker_size - min_marker_size)
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
            edgecolors='black',
            linewidth=0.5
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
    ax.text(0.02, 0.98, 'Marker size ∝ Sample size', 
            transform=ax.transAxes, fontsize=20, 
            verticalalignment='top', alpha=0.7)
    
    fig.savefig(path_dict["paper_plots"] + "informed_shares.png", dpi=300, bbox_inches='tight')
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
    initial_informed_share = params[params["parameter"] == "initial_informed_share"]["estimate"].values[0]
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