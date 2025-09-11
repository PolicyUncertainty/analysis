import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from set_styles import set_plot_defaults, set_colors


def plot_savings_rate(path_dict, specs, covariate=None, show=False, save=False, window=1):
    """
    Plot average savings rate over the life cycle (by period).
    If covariate is specified and is integer/categorical, plot for each value as well.
    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories.
    specs : dict
        Model specifications.
    covariate : str, optional
        Column name for grouping (must be integer or categorical).
    show : bool, default False
        Whether to display the plot.
    save : bool, default False
        Whether to save the plot.
    window : int, default 1
        Window size for rolling average smoothing (1 means no smoothing).
    """
    set_plot_defaults()
    colors, _ = set_colors()
    
    # Load structural estimation sample
    df = pd.read_csv(path_dict["struct_est_sample"])

    # hh net income needs to be reflated
    df["hh_net_income"] *= specs["wealth_unit"]

    # Compute average savings rate by period
    grouped = df.groupby("period")
    avg_savings = grouped["savings_dec"].mean()
    avg_income = grouped["hh_net_income"].mean()
    avg_rate = avg_savings / avg_income
    if window > 1:
        avg_rate = avg_rate.rolling(window=window, min_periods=1).mean()
    
    fig, ax = plt.subplots()
    ax.plot(avg_rate.index, avg_rate.values, label="Average", color=colors[0])
    
    # If covariate is specified and is integer/categorical, plot for each value
    if covariate is not None and covariate in df.columns:
        dtype = df[covariate].dtype
        if np.issubdtype(dtype, np.integer) or dtype == "category" or df[covariate].nunique() < 10:
            for i, val in enumerate(sorted(df[covariate].dropna().unique())):
                sub = df[df[covariate] == val]
                grouped_sub = sub.groupby("period")
                avg_savings_sub = grouped_sub["savings_dec"].mean()
                avg_income_sub = grouped_sub["hh_net_income"].mean()
                avg_rate_sub = avg_savings_sub / avg_income_sub
                if window > 1:
                    avg_rate_sub = avg_rate_sub.rolling(window=window, min_periods=1).mean()
                ax.plot(
                    avg_rate_sub.index,
                    avg_rate_sub.values,
                    label=f"{covariate}={val}",
                    color=colors[(i+1) % len(colors)]
                )
    
    ax.set_xlabel("Period")
    ax.set_ylabel("Average Savings Rate")
    #ax.set_title("Average Savings Rate over Life Cycle")
    ax.legend()
    ax.set_xlim(left=None, right=50)
    ax.set_ylim(bottom=-0.2, top=0.5)
    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["data_plots"] + "savings_rate.pdf", bbox_inches="tight")
        fig.savefig(path_dict["data_plots"] + "savings_rate.png", bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
