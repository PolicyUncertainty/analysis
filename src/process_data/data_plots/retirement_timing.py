import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import get_figsize, set_colors


def plot_retirement_timing_data(path_dict, specs, show=False, save=False):
    """Plot retirement timing relative to statutory retirement age.

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
    struct_est_sample = pd.read_csv(path_dict["struct_est_sample"])
    df_fresh = struct_est_sample[
        (struct_est_sample["choice"] == 0) & (struct_est_sample["lagged_choice"] != 0)
    ].copy()

    # Calculate actual retirement age vs SRA
    df_fresh["age"] = df_fresh["period"] + specs["start_age"]
    df_fresh["SRA"] = (
        specs["min_SRA"] + df_fresh["policy_state"] * specs["SRA_grid_size"]
    )
    df_fresh["actual_ret_age_vs_SRA"] = df_fresh["age"] - df_fresh["SRA"]

    # bar chart of actual retirement age vs SRA as percentage of total
    fig, axs = plt.subplots(2, 2, figsize=get_figsize(2, 2))
    # Plot in first plot distance and in second the age
    for sex, sex_label in enumerate(specs["sex_labels"]):
        df_sex = df_fresh[df_fresh["sex"] == sex]
        counts_distance = df_sex["actual_ret_age_vs_SRA"].value_counts().sort_index()
        counts_age = df_sex["age"].value_counts().sort_index()

        axs[sex, 0].plot(counts_distance, color=colors[sex])
        axs[sex, 0].set_title(f"Actual Retirement Age vs SRA; {sex_label}")

        axs[sex, 1].plot(counts_age, color=colors[sex])
        axs[sex, 1].set_title(f"Retirement Age; {sex_label}")

    plt.tight_layout()

    if save:
        fig.savefig(
            path_dict["data_plots"] + "retirement_timing.pdf", bbox_inches="tight"
        )
        fig.savefig(
            path_dict["data_plots"] + "retirement_timing.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_derived_retirement_age_months(
    path_dict, specs, min_age=55, max_age=72, show=False, save=False
):
    """Plot the distribution of the derived (spell-based) retirement age in months.

    Uses the cached artkalen output (art_choice.pkl), where corrected_age holds the
    retirement spell start minus the birth date for the first retirement observation.
    Fresh retirees are observations with art_choice 0 and an observed non-retirement
    choice in the previous year, within the estimation years.
    """
    colors, _ = set_colors()
    df = pd.read_pickle(path_dict["struct_data"] + "art_choice.pkl")
    df["lagged_art_choice"] = df.groupby("pid")["art_choice"].shift(1)

    syear = df.index.get_level_values("syear")
    fresh_mask = (
        (df["art_choice"] == 0)
        & df["lagged_art_choice"].isin([1, 2, 3])
        & (syear >= specs["start_year"])
        & (syear <= specs["end_year"])
    )
    ret_age = df.loc[fresh_mask, "corrected_age"].dropna()

    # Floor to full months (small epsilon guards against float noise in dates)
    ret_age_months = np.floor(ret_age * 12 + 1e-6) / 12
    shares = ret_age_months.value_counts(normalize=True).sort_index()
    shares = shares[(shares.index >= min_age) & (shares.index < max_age + 1)]

    fig, ax = plt.subplots(figsize=get_figsize(1, 1))
    ax.bar(shares.index, shares.values, width=1 / 12, align="edge", color=colors[0])
    ax.set_xticks(range(min_age, max_age + 1))
    ax.set_xlim(min_age, max_age + 1)
    ax.set_xlabel("Retirement age (monthly)")
    ax.set_ylabel("Share of fresh retirees")
    ax.set_title(f"Derived retirement age (N = {len(ret_age)})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(
            path_dict["data_plots"] + "derived_retirement_age_months.pdf",
            bbox_inches="tight",
        )
        fig.savefig(
            path_dict["data_plots"] + "derived_retirement_age_months.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
    return shares
