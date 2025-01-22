import pandas as pd
from set_paths import create_path_dict
import matplotlib.pyplot as plt
# source: https://www.deutsche-rentenversicherung.de/SharedDocs/Downloads/DE/Statistiken-und-Berichte/statistikpublikationen/rv_in_zeitreihen.html

def plot_pension_rates(paths_dict):
    df_rates = pd.read_csv(paths_dict["intermediate_data"] + "pension_payout_and_contribution_rates.csv")
    years = df_rates["year"]
    replacement_rates = df_rates["replacement_rate"]
    contribution_rates = df_rates["contribution_rate"]

    fig, ax1 = plt.subplots()

    # Plot replacement rates on the left y-axis
    ax1.plot(years, replacement_rates, label="Replacement Rate", color='C0')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Replacement Rate")
    ax1.set_ylim(45, 55)
    ax1.tick_params(axis='y')

    # Create a second y-axis for the contribution rates
    ax2 = ax1.twinx()
    ax2.plot(years, contribution_rates, label="Contribution Rate", color='C1')
    ax2.set_ylabel("Contribution Rate")
    ax2.set_ylim(15, 25)
    ax2.tick_params(axis='y')

    # Add legends
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.tight_layout()