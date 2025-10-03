import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from set_styles import set_colors

from process_data.structural_sample_scripts.policy_state import (
    create_SRA_by_gebjahr,
)


def plot_SRA_2007_reform(path_dict, show=False, save=False):
    """Plot the 2007 pension reform showing changes in statutory retirement age.
    
    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    show : bool, default False
        Whether to display plots
    save : bool, default False  
        Whether to save plots to disk
    """
    colors, _ = set_colors()
    gebjahr = pd.Series(data=np.arange(1945, 1966, 1), name="gebjahr")
    policy_states = create_SRA_by_gebjahr(gebjahr)
    policy_states_pre_reform = 65 * np.ones(gebjahr.shape[0])
    
    fig, ax = plt.subplots()
    ax.plot(gebjahr, policy_states, color=colors[0], label="post-reform")
    ax.plot(
        gebjahr,
        policy_states_pre_reform,
        linestyle="--",
        color=colors[0],
        label="pre-reform",
    )
    ax.set_xlim(1945, 1965)
    ax.set_ylim([64.8, 67.2])
    ax.set_xticks(np.arange(1945, 1966, 5))
    ax.set_yticks([65, 66, 67])
    ax.set_xlabel("Year of birth")
    ax.set_ylabel("SRA")
    ax.legend()
    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["misc_plots"] + "SRA_2007_reform.pdf", bbox_inches="tight")
        fig.savefig(path_dict["misc_plots"] + "SRA_2007_reform.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)
