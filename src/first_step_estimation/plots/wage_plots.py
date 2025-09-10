# Description: This file contains plotting functions for wage equation estimation results.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from set_styles import set_colors


def plot_wage_regression_results(path_dict, specs, show=False, save=False):
    """Plot wage regression results comparing observed vs estimated log wages by age.
    
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
    # Get colors and labels
    JET_COLOR_MAP, _ = set_colors()
    edu_labels = specs["education_labels"]
    sex_labels = specs["sex_labels"]
    file_appends = ["men", "women"]
    
    # Load wage data with predictions (this should be saved by estimation module)
    wage_data = pd.read_csv(
        path_dict["first_step_data"] + "wage_estimation_sample_with_predictions.csv", 
        index_col=0
    )
    
    # Create plots for each sex
    for sex_val, sex_label in enumerate(sex_labels):
        fig, ax = plt.subplots()
        
        for edu_val, edu_label in enumerate(edu_labels):
            # Filter data for this education-sex combination
            wage_data_type = wage_data[
                (wage_data["education"] == edu_val) & (wage_data["sex"] == sex_val)
            ].copy()
            
            if len(wage_data_type) == 0:
                continue
                
            # Plot observed log wages by age
            observed_by_age = wage_data_type.groupby("age")["ln_wage"].mean()
            ax.plot(
                observed_by_age.index,
                observed_by_age.values,
                color=JET_COLOR_MAP[edu_val],
                ls="--",
                label=f"Obs. {edu_label}",
            )
            
            # Plot predicted log wages by age
            predicted_by_age = wage_data_type.groupby("age")["predicted_ln_wage"].mean()
            ax.plot(
                predicted_by_age.index,
                predicted_by_age.values,
                color=JET_COLOR_MAP[edu_val],
                label=f"Est. {edu_label}",
            )
            
        ax.set_title(f"{sex_label}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Log hourly wage")
        ax.legend(loc="upper left")
        
        if save:
            plt.savefig(
                path_dict["first_step_plots"] + f"wages_{file_appends[sex_val]}.png",
                bbox_inches="tight"
            )
    
    if show:
        plt.show()