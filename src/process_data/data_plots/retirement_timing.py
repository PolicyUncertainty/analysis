import matplotlib.pyplot as plt
import pandas as pd
from set_styles import set_colors, get_figsize


def plot_retirement_timing_data(path_dict, specs, show=False, save=False):
    """Plot retirement timing relative to statutory retirement age.
    
    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    specs : dict
        Dictionary containing model specifications
    show : bool, default False
        Whether to display plots
    save : bool, default False  
        Whether to save plots to disk
    """
    colors, _ = set_colors()
    struct_est_sample = pd.read_csv(path_dict["struct_est_sample"])
    df_fresh = struct_est_sample[
        (struct_est_sample["choice"] == 0) & (struct_est_sample["lagged_choice"] != 0)
    ].copy()

    # Calculate actual retirement age vs SRA
    df_fresh["age"] = df_fresh["period"] + specs["start_age"]
    df_fresh["SRA"] = (
        specs["min_SRA"] + df_fresh["policy_state"] * specs["SRA_grid_size"]
    )
    df_fresh["actual_ret_age_vs_SRA"] = df_fresh["age"] - df_fresh["SRA"]

    # bar chart of actual retirement age vs SRA as percentage of total
    fig, axs = plt.subplots(2, 2, figsize=get_figsize(2, 2))
    # Plot in first plot distance and in second the age
    for sex, sex_label in enumerate(specs["sex_labels"]):
        df_sex = df_fresh[df_fresh["sex"] == sex]
        counts_distance = df_sex["actual_ret_age_vs_SRA"].value_counts().sort_index()
        counts_age = df_sex["age"].value_counts().sort_index()

        axs[sex, 0].plot(counts_distance, color=colors[sex])
        axs[sex, 0].set_title(f"Actual Retirement Age vs SRA; {sex_label}")

        axs[sex, 1].plot(counts_age, color=colors[sex])
        axs[sex, 1].set_title(f"Retirement Age; {sex_label}")

    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["data_plots"] + "retirement_timing.pdf", bbox_inches="tight")
        fig.savefig(path_dict["data_plots"] + "retirement_timing.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)
