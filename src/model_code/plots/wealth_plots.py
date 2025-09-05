# Description: This file contains plotting functions for wealth and budget model results.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from set_styles import set_colors
from model_code.wealth_and_budget.budget_equation import budget_constraint


def plot_budget_of_unemployed(path_dict, specs, show=False, save=False):
    """Plot the budget constraint (=wealth) for different levels of end-of period
    savings of an unemployed person.

    Special emphasis on the area around eligibility for unemployment benefits.
    
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
    savings = np.linspace(0, 5, 50)

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        ax = axs[sex_var]
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            wealth, _ = budget_constraint(
                period=70,
                education=edu_var,
                lagged_choice=1,
                experience=0.01,
                sex=sex_var,
                partner_state=np.array([0]),
                asset_end_of_previous_period=savings,
                income_shock_previous_period=0,
                model_specs=specs,
            )
            ax.plot(savings, wealth, label=edu_label, color=colors[edu_var])
            
        ax.set_xlabel("Savings")
        ax.set_ylabel("Wealth")
        ax.legend()
        ax.set_title(f"Unemployment benefits {sex_label}")
    
    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["model_plots"] + "wealth_unemployed_budget.pdf", bbox_inches="tight")
        fig.savefig(path_dict["model_plots"] + "wealth_unemployed_budget.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)