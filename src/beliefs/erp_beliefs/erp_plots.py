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

    for edu_val, edu_label in enumerate(specs["education_labels"]):
        # Filter data for the current education level
        df_restricted = df[df["education"] == edu_val]
        params_edu = params[params["type"] == edu_label]
        
        # Generate observed shares and weights
        observed_shares_edu, weights = generate_observed_informed_shares(df_restricted)
        
        # Generate predicted shares
        predicted_shares_edu = predicted_shares_by_age(
            params=params_edu, ages_to_predict=ages_to_predict
        )
    
        # Update the DataFrames with the results for the current education level
        observed_shares[edu_label] = observed_shares_edu
        predicted_shares[edu_label] = predicted_shares_edu
    
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
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        ax.plot(
            observed_shares[edu_label].rolling(window=3).mean(),
            label=f"Obs. {edu_label}",
            marker="o",
            linestyle="None",
            markersize=4,
            color=JET_COLOR_MAP[edu_val],
        )
        ax.plot(
            predicted_shares[edu_label],
            color=JET_COLOR_MAP[edu_val],
            label=f"Est. {edu_label}",
        )
    # Set labels
    ax.set_xlabel("Age")
    ax.set_ylabel("Share Informed")
    ax.legend()
    fig.savefig(path_dict["paper_plots"] + "informed_shares.png")
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
    return informed_by_age, weights



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