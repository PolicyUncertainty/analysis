import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from export_results.figures.color_map import JET_COLOR_MAP
from process_data.structural_sample_scripts.policy_state import (
    create_SRA_by_gebjahr,
)


def plot_SRA_2007_reform(path_dict):
    gebjahr = pd.Series(data=np.arange(1945, 1966, 1), name="gebjahr")
    policy_states = create_SRA_by_gebjahr(gebjahr)
    policy_states_pre_reform = 65 * np.ones(gebjahr.shape[0])
    fig, ax = plt.subplots()
    ax.plot(gebjahr, policy_states, color="C0", label="post-reform")
    ax.plot(
        gebjahr,
        policy_states_pre_reform,
        linestyle="--",
        color="C0",
        label="pre-reform",
    )
    ax.set_xlim(1945, 1965)
    ax.set_ylim([64.8, 67.2])
    ax.set_xticks(np.arange(1945, 1966, 5))
    ax.set_yticks([65, 66, 67])
    ax.set_xlabel("Year of birth")
    ax.set_ylabel("SRA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_dict["plots"] + "introduction/SRA_2007_reform.png", transparent=True, dpi=300)


def plot_SRA_2007_reform_extended(path_dict, ref_year=2025, end_year=1995):
    """
    Extended version of SRA 2007 reform plot with extended x-axis and CEE proposal.
    
    Parameters:
    -----------
    path_dict : dict
        Dictionary containing path to plots directory
    ref_year : int or None, default=2025
        Reference year for age calculation on second x-axis. 
        If None, second x-axis is not shown.
    end_year : int, default=1995
        End year for birth cohort range
    """
    
    # Create extended birth year range
    gebjahr = pd.Series(data=np.arange(1945, end_year + 1, 1), name="gebjahr")
    
    # Get policy states for the extended range
    policy_states = create_SRA_by_gebjahr(gebjahr)
    policy_states_pre_reform = 65 * np.ones(gebjahr.shape[0])
    
    # Create CEE proposal values for birth cohorts after 1964
    cee_proposal = np.full(gebjahr.shape[0], np.nan)
    cee_mask = gebjahr > 1964
    cee_proposal[cee_mask] = 67 + 0.05 * (gebjahr[cee_mask] - 1964)
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots()
    
    # Plot the three lines in correct legend order
    line1 = ax1.plot(
        gebjahr,
        policy_states_pre_reform,
        linestyle="--",
        color=JET_COLOR_MAP[0],
        label="pre 2007 reform",
    )
    line2 = ax1.plot(gebjahr, policy_states, color=JET_COLOR_MAP[0], label="post 2007 reform")
    line3 = ax1.plot(
        gebjahr,
        cee_proposal,
        linestyle=":",
        color=JET_COLOR_MAP[1],
        label="CEE proposal"
    )
    
    # Add red question mark at intersection point (1964, 67)
    ax1.text(1964.25, 67.2, '?', fontsize=30, color=JET_COLOR_MAP[3], 
             ha='center', va='center')
    
    # Set primary axis properties
    ax1.set_xlim(1945, end_year)
    ax1.set_ylim([64.8, 69])
    ax1.set_xticks(np.arange(1945, end_year + 1, 10))
    ax1.set_yticks(np.arange(65, 70, 1))
    ax1.set_xlabel("Year of birth", labelpad=10)
    ax1.set_ylabel("Statutory Retirement Age", labelpad=10)
    ax1.legend()
    
    # Add second x-axis showing age in reference year if ref_year is provided
    if ref_year is not None:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        
        # Create age ticks - show ages corresponding to birth year ticks
        birth_year_ticks = ax1.get_xticks()
        age_ticks = ref_year - birth_year_ticks
        ax2.set_xticks(birth_year_ticks)
        ax2.set_xticklabels([f"{int(age)}" for age in age_ticks])
        ax2.set_xlabel(f"Age in {ref_year}", labelpad=10)
    
    fig.tight_layout()
    fig.savefig(path_dict["plots"] + "introduction/SRA_2007_reform_extended.png", transparent=True, dpi=300)


# minimal working example 
if __name__ == "__main__":
    from set_paths import create_path_dict
    import os
    path_dict = create_path_dict(define_user=False)
    os.makedirs(path_dict["plots"] + "introduction", exist_ok=True)
    plot_SRA_2007_reform(path_dict)
    plot_SRA_2007_reform_extended(path_dict, ref_year=None, end_year=1995)
