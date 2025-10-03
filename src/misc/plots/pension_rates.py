import pandas as pd
import matplotlib.pyplot as plt
from set_styles import set_colors
# source: https://www.deutsche-rentenversicherung.de/SharedDocs/Downloads/DE/Statistiken-und-Berichte/statistikpublikationen/rv_in_zeitreihen.html


def plot_pension_rates(path_dict, show=False, save=False):
    """Plot pension replacement and contribution rates over time.
    
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
    df_rates = pd.read_csv(path_dict["open_data"] + "pension_payout_and_contribution_rates.csv")
    years = df_rates["year"]
    replacement_rates = df_rates["replacement_rate"]
    contribution_rates = df_rates["contribution_rate"]

    fig, ax1 = plt.subplots()

    # Plot replacement rates on the left y-axis
    ax1.plot(years, replacement_rates, label="Replacement Rate", color=colors[0])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Replacement Rate")
    ax1.set_ylim(45, 55)
    ax1.tick_params(axis='y')

    # Create a second y-axis for the contribution rates
    ax2 = ax1.twinx()
    ax2.plot(years, contribution_rates, label="Contribution Rate", color=colors[1])
    ax2.set_ylabel("Contribution Rate")
    ax2.set_ylim(15, 25)
    ax2.tick_params(axis='y')

    # Add legends
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["misc_plots"] + "pension_rates.pdf", bbox_inches="tight")
        fig.savefig(path_dict["misc_plots"] + "pension_rates.png", bbox_inches="tight")
        
    if show:
        plt.show()
    else:
        plt.close(fig)