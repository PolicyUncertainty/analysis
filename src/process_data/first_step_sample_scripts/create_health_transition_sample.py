# %%
import os

import pandas as pd
import numpy as np
from process_data.aux_scripts.filter_data import filter_below_age
from process_data.aux_scripts.filter_data import filter_above_age
from process_data.aux_scripts.filter_data import filter_years
from process_data.aux_scripts.filter_data import filter_by_sex
from process_data.soep_vars.education import create_education_type
from process_data.soep_vars.health import create_health_state
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

    df = create_education_type(df)

    # Filter estimation years
    df = filter_years(df, specs["start_year"], specs["end_year"])

    # lagged health state and health state
    df = create_health_and_lagged_state(df, specs)

    # Filter age and sex
    df = filter_below_age(df, specs["start_age"])
    df = filter_above_age(df, specs["end_age"])
    df = filter_by_sex(df, no_women=False)



    df = df[
        ["age", "sex", "education", "health_state", "lagged_health_state"]
    ]

    print(
        str(len(df))
        + " observations in the final health transition sample.  \n ----------------"
    )
    
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.kernel_regression import KernelReg

    # Define the window size for the moving average
    windowsize = 5

    # Compute symmetric moving average for mean health state by age for each education level (fancy means)
    health_h = df[df["education"] == 1].groupby("age")["health_state"].mean().rolling(windowsize, center=True).mean()
    health_l = df[df["education"] == 0].groupby("age")["health_state"].mean().rolling(windowsize, center=True).mean()

    # Aggregate the data to compute mean health state for each age and education level
    mean_df = (
        df.groupby(["age", "education"])["health_state"]
        .mean()
        .reset_index()
    )

    # Create the plot
    fig, ax = plt.subplots()

    # Define more visually appealing red and blue colors
    colors = {0: "#D72638", 1: "#1E90FF"}  # Red: #D72638, Blue: #1E90FF

    # Plot mean values for each education level as scatter points only
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

    # Calculate the relative frequency of good health for each education level and lagged health state combination
    hgg_h = df[(df["education"] == 1) & (df["lagged_health_state"] == 1)].groupby("age")["health_state"].mean() # good health -> good health, high education
    hgg_l = df[(df["education"] == 0) & (df["lagged_health_state"] == 1)].groupby("age")["health_state"].mean() # good health -> good health, low education
    hbg_h = df[(df["education"] == 1) & (df["lagged_health_state"] == 0)].groupby("age")["health_state"].mean() # bad health -> good health, high education
    hbg_l = df[(df["education"] == 0) & (df["lagged_health_state"] == 0)].groupby("age")["health_state"].mean() # bad health -> good health, low education

    # Calculate complementary bad health shock probabilities 
    hgb_h = 1 - hgg_h                                                                                           # good health -> bad health, high education
    hgb_l = 1 - hgg_l                                                                                           # good health -> bad health, low education                 

    # Get the symmetric moving average for each education level for hbg_l, hbg_h, hgb_l, hgb_h across age groups
    hbg_h_m = hbg_h.rolling(windowsize, center=True).mean()
    hbg_l_m = hbg_l.rolling(windowsize, center=True).mean()
    hgb_h_m = hgb_h.rolling(windowsize, center=True).mean()
    hgb_l_m = hgb_l.rolling(windowsize, center=True).mean()

    # Create the figures
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Panel (a): Probability of good health shock
    axs[0].plot(hbg_l_m.index, hbg_l_m, color=colors[0], label=f"Low education (MA, ws={windowsize})")
    axs[0].scatter(hbg_l.index, hbg_l, color=colors[0], alpha=0.65, label="Low education")
    axs[0].plot(hbg_h_m.index, hbg_h_m, color=colors[1], label=f"High education (MA, ws={windowsize})")
    axs[0].scatter(hbg_h.index, hbg_h, color=colors[1], alpha=0.65, label="High education")
    axs[0].set_title("Probability of Good Health Shock (relative frequency)", fontsize=14)
    axs[0].set_ylabel("Probability", fontsize=12)
    axs[0].set_xlabel("Age (years)", fontsize=12)
    axs[0].legend(loc="upper right")
    axs[0].set_ylim(0, 0.9)
    axs[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axs[1].set_xticks([30, 40, 50, 60, 70, 80])
    axs[0].grid(False)

    # Panel (b): Probability of bad health shock
    axs[1].plot(hgb_l_m.index, hgb_l_m, color=colors[0], label=f"Low education (MA, ws={windowsize})")
    axs[1].scatter(hgb_l.index, hgb_l, color=colors[0], alpha=0.65, label="Low education")
    axs[1].plot(hgb_h_m.index, hgb_h_m, color=colors[1], label=f"High education (MA, ws={windowsize})")
    axs[1].scatter(hgb_h.index, hgb_h, color=colors[1], alpha=0.65, label="High education")
    axs[1].set_title("Probability of Bad Health Shock (relative frequency)", fontsize=14)
    axs[1].set_xlabel("Age (years)", fontsize=12)
    axs[1].set_ylabel("Probability", fontsize=12)
    axs[1].legend(loc="upper right")
    axs[1].set_ylim(0, 0.2)
    axs[1].set_yticks([0, 0.1, 0.2])
    axs[1].set_xticks([30, 40, 50, 60, 70, 80])
    axs[1].grid(False)

    # Display the plots
    plt.show()




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
    merged_data.rename(columns={"m11126": "srh", "m11124": "disabil" }, inplace=True)
    merged_data["age"] = merged_data["syear"] - merged_data["gebjahr"]
    merged_data.set_index(["pid", "syear"], inplace=True)
    print(str(len(merged_data)) + " observations in SOEP C38 core.")
    return merged_data

def create_health_and_lagged_state(df, specs):
    # The following code is dependent on span dataframe being called first.
    # In particular the lagged partner state must be after span dataframe and create partner state.
    # We should rewrite this
    df = clean_health(df)
    df = span_dataframe(df, specs)
    df = create_health_state(df)
    df["lagged_health_state"] = df.groupby(["pid"])["health_state"].shift()
    df = df[df["lagged_health_state"].notna()]
    df = df[df["health_state"].notna()]

    return df


