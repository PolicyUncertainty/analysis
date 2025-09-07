import numpy as np
from matplotlib import pyplot as plt
from set_styles import set_colors

from model_code.state_space.experience import (
    construct_experience_years,
    get_next_period_experience,
    scale_experience_years,
)


def plot_ret_experience(path_dict, specs, show=False, save=False):
    """Plot retirement experience law of motion.
    
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
    periods = np.arange(30, 40)
    fig, ax = plt.subplots()
    
    for i, exp_years in enumerate(np.arange(30, 50, 10)):
        # We scale the experience for a not retired person last period (policy state 8 is below)
        exp = scale_experience_years(
            experience_years=exp_years,
            period=periods - 1,
            is_retired=np.zeros_like(periods, dtype=bool),
            model_specs=specs,
        )

        exp_next = get_next_period_experience(
            period=periods,
            lagged_choice=0,
            policy_state=8,
            sex=0,
            education=1,
            experience=exp,
            informed=1,
            health=0,
            model_specs=specs,
        )
        exp_years_next = construct_experience_years(
            float_experience=exp_next,
            period=periods,
            is_retired=np.ones_like(periods, dtype=bool),
            model_specs=specs,
        )
        ax.plot(periods + 30, exp_years_next, label=f"Exp {exp_years}", color=colors[i % len(colors)])

    ax.legend()
    ax.set_xlabel("Period")
    ax.set_ylabel("Experience Years")
    ax.set_title("Retirement Experience Law of Motion")
    
    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["model_plots"] + "retirement_experience_law_of_motion.pdf", bbox_inches="tight")
        fig.savefig(path_dict["model_plots"] + "retirement_experience_law_of_motion.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)
