import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from set_styles import set_colors
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_pension_claims(path_dict, show=False, save=True):
    """Plot stacked bar chart of pension claims by type over time."""
    specs = generate_derived_and_data_derived_specs(path_dict)
    
    # Load data
    df_pension_claims = pd.read_csv(
        path_dict["open_data"] + "rentenzugaenge/rentenzugang_aggregiert_geschlecht_jahr.csv", 
        sep=","
    )
    
    # Rename columns and sex values
    rename_dict = {
        "gevs": "sex",
        "ztptrtbe_jjjj": "year",
        "rente_regel": "old_age_standard",
        "rente_beslang": "old_age_long_working_life",
        "rente_lang": "old_age_with_penalties",
        "rente_em": "disability",
        "rente_schwerb": "disability_severe",
        "rente_frau": "women",
        "rente_alo": "unemployment",
        "rente_andere": "other_pension"
    }
    
    rename_dict_sex = {
        "männlich": "male",
        "weiblich": "female"
    }
    
    df_pension_claims = df_pension_claims.rename(columns=rename_dict)
    df_pension_claims["sex"] = df_pension_claims["sex"].replace(rename_dict_sex)
    
    # Filter years and fill NaN
    start_year = specs["start_year"]
    end_year = specs["end_year"]
    years = np.arange(start_year, end_year + 1)
    
    df_plot = df_pension_claims[df_pension_claims["year"].isin(years)].copy().fillna(0)
    
    # Combine other categories
    df_plot["other"] = df_plot["women"] + df_plot["unemployment"] + df_plot["other_pension"]
    df_plot = df_plot.drop(columns=["women", "unemployment", "other_pension"])
    
    # Collapse by sex
    df_plot = df_plot.groupby("year", as_index=False).sum(numeric_only=True)
    
    # Get colors
    color_map, _ = set_colors()
    
    # Define custom colors for categories
    colors = {
        "old_age_standard": color_map[0],      # Blue family for old age pensions
        "old_age_with_penalties": color_map[1], # Orange family 
        "old_age_long_working_life": color_map[2], # Green family
        "disability": color_map[3],            # Red family for disability
        "disability_severe": color_map[4],     # Purple family
        "other": "#7f7f7f"                     # Gray for other
    }
    
    # Prepare data for plotting
    cols_to_plot = ["old_age_standard", "old_age_with_penalties", "old_age_long_working_life",
                   "disability", "disability_severe", "other"]
    
    # Create labels with proper formatting
    labels = [col.replace("_", ": ") for col in cols_to_plot]
    
    # Create the plot
    fig, ax = plt.subplots()
    
    bottom = np.zeros(len(df_plot))
    for i, col in enumerate(cols_to_plot):
        ax.bar(df_plot["year"], df_plot[col], bottom=bottom, 
               color=colors[col], label=labels[i], alpha=0.8)
        bottom += df_plot[col]
    
    ax.set_ylabel("Number of pension claims")
    ax.set_xlabel("Year")
    ax.set_title("Pension Claims by Type")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    
    if save:
        plt.savefig(path_dict["misc_plots"] + "pension_claims.png", bbox_inches="tight")
    if show:
        plt.show()
    
    plt.close()