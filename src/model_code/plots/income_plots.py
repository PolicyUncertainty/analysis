# Description: This file contains plotting functions for income model results.
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from set_styles import set_colors
from model_code.pension_system.experience_stock import (
    calc_pension_points_for_experience,
)
from model_code.state_space.experience import scale_experience_years
from model_code.wealth_and_budget.budget_equation import budget_constraint
from model_code.wealth_and_budget.pension_payments import (
    calc_gross_pension_income,
    calc_pensions_after_ssc,
)
from model_code.wealth_and_budget.transfers import calc_child_benefits
from model_code.wealth_and_budget.wages import (
    calc_labor_income_after_ssc,
    calculate_gross_labor_income,
)
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_incomes(path_dict, show=False, save=False):
    """Plots incomes by experience level for men and women.
    
    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    show : bool, default False
        Whether to display plots
    save : bool, default False  
        Whether to save plots to disk
    """
    specs = generate_derived_and_data_derived_specs(path_dict)
    colors, _ = set_colors()
    
    exp_levels = np.arange(0, 50)

    annual_unemployment = specs["annual_unemployment_benefits"]
    unemployment_benefits = np.ones_like(exp_levels) * annual_unemployment

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        # Now loop over education to generate specific net and gross wages and pensions
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axes[sex_var, edu_var]

            ax.plot(
                exp_levels,
                unemployment_benefits,
                label="Unemployment benefits",
                color=colors[0],
            )

            # Initialize empty containers, part and full time wages and pensions
            gross_pt_wages = np.zeros_like(exp_levels, dtype=float)
            after_ssc_pt_wages = np.zeros_like(exp_levels, dtype=float)
            gross_ft_wages = np.zeros_like(exp_levels, dtype=float)
            after_ssc_ft_wages = np.zeros_like(exp_levels, dtype=float)
            gross_pensions = np.zeros_like(exp_levels, dtype=float)
            after_ssc_pensions = np.zeros_like(exp_levels, dtype=float)

            for exp_idx, experience in enumerate(exp_levels):
                # Calculate part-time wages (lagged_choice=2)
                gross_pt_wages[exp_idx] = calculate_gross_labor_income(
                    lagged_choice=2,
                    experience_years=experience,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )
                after_ssc_pt_wages[exp_idx], _ = calc_labor_income_after_ssc(
                    lagged_choice=2,
                    experience_years=experience,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )

                # Calculate full-time wages (lagged_choice=3)
                gross_ft_wages[exp_idx] = calculate_gross_labor_income(
                    lagged_choice=3,
                    experience_years=experience,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )
                after_ssc_ft_wages[exp_idx], _ = calc_labor_income_after_ssc(
                    lagged_choice=3,
                    experience_years=experience,
                    education=edu_var,
                    sex=sex_var,
                    income_shock=0,
                    model_specs=specs,
                )

                # Calculate pensions
                exp_stock_pension = calc_pension_points_for_experience(
                    period=37,
                    sex=sex_var,
                    experience_years=experience,
                    education=edu_var,
                    policy_state=8,  # assume informed policy state
                    informed=True,
                    health=1,  # assume good health
                    model_specs=specs,
                )
                gross_pensions[exp_idx] = calc_gross_pension_income(
                    exp_stock_pension, specs
                )
                after_ssc_pensions[exp_idx] = calc_pensions_after_ssc(
                    gross_pensions[exp_idx], specs
                )

            # Plot wages and pensions
            ax.plot(exp_levels, gross_pt_wages, label="Gross PT wages", color=colors[1])
            ax.plot(exp_levels, after_ssc_pt_wages, label="Net PT wages", color=colors[2])
            ax.plot(exp_levels, gross_ft_wages, label="Gross FT wages", color=colors[3])
            ax.plot(exp_levels, after_ssc_ft_wages, label="Net FT wages", color=colors[4])
            ax.plot(exp_levels, gross_pensions, label="Gross pensions", color=colors[5])
            ax.plot(exp_levels, after_ssc_pensions, label="Net pensions", color=colors[6])

            ax.set_title(f"{sex_label}, {edu_label}")
            ax.set_xlabel("Experience (years)")
            ax.set_ylabel("Annual Income (€)")
            
            # Only show legend on first subplot
            if sex_var == 0 and edu_var == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["model_plots"] + "income_by_experience.pdf", bbox_inches="tight")
        fig.savefig(path_dict["model_plots"] + "income_by_experience.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)