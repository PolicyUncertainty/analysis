# Description: This file contains plotting functions for partner wage equation estimation results.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from set_styles import set_colors, get_figsize


def plot_partner_wage_results(path_dict, specs, show=False, save=False):
    """Plot partner wage estimation results comparing observed vs estimated wages by age.
    
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
    
    # Load partner wage data with predictions (this should be saved by estimation module)
    wage_data = pd.read_csv(
        path_dict["first_step_data"] + "partner_wage_estimation_sample_with_predictions.csv", 
        index_col=0
    )

    # Create plots for each sex
    for sex_val, sex_label in enumerate(sex_labels):
        fig, ax = plt.subplots()
        
        for edu_val, edu_label in enumerate(edu_labels):
            # Filter data for this education-sex combination
            wage_data_edu = wage_data[
                (wage_data["education"] == edu_val) & (wage_data["sex"] == sex_val)
            ].copy()
            
            if len(wage_data_edu) == 0:
                continue
                
            # Plot observed wages by age
            observed_by_age = wage_data_edu.groupby("age")["wage_p"].mean()
            ax.plot(
                observed_by_age.index,
                observed_by_age.values,
                label=f"Obs. {edu_label}",
                linestyle="--",
                color=JET_COLOR_MAP[edu_val],
            )
            
            # Plot predicted wages by age
            predicted_by_age = wage_data_edu.groupby("age")["wage_pred"].mean()
            ax.plot(
                predicted_by_age.index,
                predicted_by_age.values,
                label=f"Est. {edu_label}",
                color=JET_COLOR_MAP[edu_val],
            )
            
        ax.legend()
        ax.set_title(f"Partner Wages of {sex_label}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Monthly Wage")
        
        if save:
            fig.savefig(
                path_dict["first_step_plots"] + f"partner_wages_{file_appends[sex_val]}.png",
                bbox_inches="tight"
            )
            fig.savefig(
                path_dict["first_step_plots"] + f"partner_wages_{file_appends[sex_val]}.pdf",
                bbox_inches="tight"
            )

    if show:
        plt.show()