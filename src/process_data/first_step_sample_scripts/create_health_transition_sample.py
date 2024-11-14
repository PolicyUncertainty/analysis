# %%
import os

import pandas as pd
import numpy as np
from process_data.aux_scripts.filter_data import filter_below_age
from process_data.aux_scripts.filter_data import filter_above_age
from process_data.aux_scripts.filter_data import filter_years
from process_data.aux_scripts.filter_data import filter_by_sex
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.health import create_health_states
from process_data.soep_vars.health import clean_health
from process_data.aux_scripts.lagged_and_lead_vars import span_dataframe



# %%
def create_health_transition_sample(paths, specs, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = (
        paths["intermediate_data"] + "health_transition_estimation_sample.pkl"
    )

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data

    df = load_and_merge_soep_core(paths["soep_c38"])

    # Pre-Filter estimation years
    df = filter_years(df, specs["start_year"] - 11, specs["end_year"] + 3)

    # Pre-Filter age and sex
    df = filter_below_age(df, specs["start_age"] - 11)
    df = filter_above_age(df, specs["end_age"] + 11)
    df = filter_by_sex(df, no_women=False)

    # Create education type
    df = create_education_type(df)

    # lagged health state and health state
    df = create_health_and_lagged_states(df, specs)

    df = df[
        ["age", "sex", "education", "health_state", "lagged_health_state"]
    ]

    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.kernel_regression import KernelReg

    # Define the window size for the moving average
    windowsize = 5




    # Define the Epanechnikov kernel function
    def epanechnikov_kernel(distance, bandwidth):
        u = distance / bandwidth
        return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

    # Function to calculate the weighted mean using the Epanechnikov kernel
    def kernel_weighted_mean(df, target_age, bandwidth):
        # Calculate the age distance and kernel weights
        age_distances = np.abs(df['age'] - target_age)
        weights = epanechnikov_kernel(age_distances, bandwidth)
        
        # Calculate the weighted mean of health states
        weighted_mean = np.sum(weights * df['health_state']) / np.sum(weights)
        return weighted_mean

    # Parameters
    bandwidth = 5  # Window size for age range smoothing
    ages = np.arange(31, 86)  # Age range from 31 to 85

    # Calculate the smoothed probabilities for each education level and health transition
    def calculate_smoothed_probabilities(education, lagged_health_state):
        smoothed_values = []
        for age in ages:
            # Filter the data for the specified education and lagged health state
            subset = df[(df['education'] == education) & (df['lagged_health_state'] == lagged_health_state)]
            smoothed_value = kernel_weighted_mean(subset, age, bandwidth)
            smoothed_values.append(smoothed_value)
        return pd.Series(smoothed_values, index=ages)

    # Calculate the relative frequency of good health for each education level by lagged health states
    hgg_h = calculate_smoothed_probabilities(education=1, lagged_health_state=1)  # High education, good -> good
    hgg_l = calculate_smoothed_probabilities(education=0, lagged_health_state=1)  # Low education, good -> good
    hbg_h = calculate_smoothed_probabilities(education=1, lagged_health_state=0)  # High education, bad -> good
    hbg_l = calculate_smoothed_probabilities(education=0, lagged_health_state=0)  # Low education, bad -> good

    # Calculate complementary bad health shock probabilities
    hgb_h = 1 - hgg_h  # High education, good -> bad
    hgb_l = 1 - hgg_l  # Low education, good -> bad


    # Filter estimation years
    df = filter_years(df, specs["start_year"], specs["end_year"])

    # Filter age and sex
    df = filter_below_age(df, specs["start_age"])
    df = filter_above_age(df, specs["end_age"])

    df = filter_by_sex(df, no_women=False)


    # Create mean health by age plot
    fig, ax = plt.subplots()

    # Define more visually appealing red and blue colors
    colors = {0: "#D72638", 1: "#1E90FF"}  # Red: #D72638, Blue: #1E90FF

    # Plot mean values for each education level as scatter points only

    # Aggregate the data to compute mean health state for each age and education level
    mean_df = (
        df.groupby(["age", "education"])["health_state"]
        .mean()
        .reset_index()
    )

    # Compute symmetric moving average for mean health state by age for each education level (fancy means)
    health_h = df[df["education"] == 1].groupby("age")["health_state"].mean().rolling(windowsize, center=True).mean()
    health_l = df[df["education"] == 0].groupby("age")["health_state"].mean().rolling(windowsize, center=True).mean()

    for edu in [0, 1]:
        edu_df = mean_df[mean_df["education"] == edu]
        txt = "Low" if edu == 0 else "High"
        
        # Plot scatter points
        ax.scatter(
            edu_df["age"],
            edu_df["health_state"],
            color=colors[edu],
            label=f"Education {txt}",
            alpha=0.65
        )

    # Plot moving average for each education level by age with window size in the label
    ax.plot(health_h.index, health_h, color=colors[1], label=f"Education High (MA, ws={windowsize})")
    ax.plot(health_l.index, health_l, color=colors[0], label=f"Education Low (MA, ws={windowsize})")

    ax.set_xlabel("Age")
    ax.set_ylabel("Mean Health State")
    ax.legend()
    plt.show()

    

    # Create the figures for the health transition probabilities
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Panel (a): Probability of good health shock
    axs[0].plot(ages, hbg_l, color=colors[1], label="Low education (smoothed)")
    axs[0].scatter(ages, hbg_l, color=colors[1], alpha=0.65, label="Low education")
    axs[0].plot(ages, hbg_h, color=colors[0], label="High education (smoothed)")
    axs[0].scatter(ages, hbg_h, color=colors[0], alpha=0.65, label="High education")
    axs[0].set_title(f"Probability of Good Health Shock (kernel-smoothed), bw={bandwidth}", fontsize=14)
    axs[0].set_ylabel("Probability", fontsize=12)
    axs[0].set_xlabel("Age (years)", fontsize=12)
    axs[0].legend(loc="upper right")
    axs[0].set_ylim(0, 0.6)
    axs[0].set_yticks(np.arange(0, 0.7, 0.1))
    axs[0].set_xticks(np.arange(30, 90, 10))
    axs[0].grid(False)

    # Panel (b): Probability of bad health shock
    axs[1].plot(ages, hgb_l, color=colors[1], label="Low education (smoothed)")
    axs[1].scatter(ages, hgb_l, color=colors[1], alpha=0.65, label="Low education")
    axs[1].plot(ages, hgb_h, color=colors[0], label="High education (smoothed)")
    axs[1].scatter(ages, hgb_h, color=colors[0], alpha=0.65, label="High education")
    axs[1].set_title(f"Probability of Bad Health Shock (kernel-smoothed), bw={bandwidth}", fontsize=14)
    axs[1].set_xlabel("Age (years)", fontsize=12)
    axs[1].set_ylabel("Probability", fontsize=12)
    axs[1].legend(loc="upper right")
    axs[1].set_ylim(0, 0.6)
    axs[1].set_yticks(np.arange(0, 0.7, 0.1))
    axs[1].set_xticks(np.arange(30, 90, 10))
    axs[1].grid(False)

    # Display the plots
    plt.show()



    print(
        str(len(df))
        + " observations in the final health transition sample.  \n ----------------"
    )
    
    



    df.to_pickle(out_file_path)
    return df


def load_and_merge_soep_core(soep_c38_path):
    # Load SOEP core data
    pgen_data = pd.read_stata(
        f"{soep_c38_path}/pgen.dta",
        columns=[
            "syear",
            "pid",
            "hid",
            "pgemplst",
            "pgpsbil",
            "pgstib",
        ],
        convert_categoricals=False,
    )
    ppathl_data = pd.read_stata(
        f"{soep_c38_path}/ppathl.dta",
        columns=["syear", "pid", "hid", "sex", "parid", "gebjahr"],
        convert_categoricals=False,
    )
    pequiv_data = pd.read_stata(
        # m11126: Self-Rated Health Status
        # m11124: Disability Status of Individual 
        f"{soep_c38_path}/pequiv.dta",
        columns=["pid", "syear", "m11126", "m11124"],
        convert_categoricals=False,
    )
    merged_data = pd.merge(
        pgen_data, ppathl_data, on=["pid", "hid", "syear"], how="inner"
    )
    merged_data = pd.merge(merged_data, pequiv_data, on=["pid", "syear"], how="inner")
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data

def create_health_and_lagged_states(df, specs):
    # The following code is dependent on span dataframe being called first.
    # In particular the lagged partner state must be after span dataframe and create partner state.
    # We should rewrite this
    df = clean_health(df)
    df = span_dataframe(df, specs)
    df = create_health_states(df)

    return df


