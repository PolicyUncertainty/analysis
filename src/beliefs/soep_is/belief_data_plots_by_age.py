import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import set_colors


def plot_erp_beliefs_by_age(paths_dict, show=False, save=False):
    JET_COLOR_MAP, LINE_STYLES = set_colors()
    # Load and prepare the data
    df_soep_is = pd.read_csv(
        paths_dict["beliefs_data"] + "soep_is_clean.csv",
        dtype={"gebjahr": int},
    )
    relevant_columns = [
        "belief_pens_deduct",
        "belief_pens_deduct_rob_times1_5",
        "belief_pens_deduct_rob_times0_5",
        "age",
    ]
    data_deduction = df_soep_is[~df_soep_is["belief_pens_deduct"].isnull()][
        relevant_columns
    ]
    age_bins = list(range(25, 66, 5))
    data_deduction["age_group"] = create_age_groups(data_deduction, age_bins=age_bins)
    ded_data_grouped = data_deduction.groupby(["age_group"], observed=True)
    ded_data_mean = ded_data_grouped["belief_pens_deduct"].mean()
    ded_data_sem = ded_data_grouped["belief_pens_deduct"].sem()
    ded_data_median = ded_data_grouped["belief_pens_deduct"].median()

    # Plot the results
    fig, ax = plt.subplots()
    ded_data_mean.plot(
        y="belief_pens_deduct",
        ax=ax,
        label="mean ERP belief",
    )
    ded_data_median.plot(
        y="belief_pens_deduct",
        ax=ax,
        ls="--",
        label="median ERP belief",
    )
    ax.errorbar(
        x=ded_data_mean.index,
        y=ded_data_mean,
        yerr=ded_data_sem,
        fmt="o",
        color="black",
        ecolor="grey",
        capsize=5,
    )
    # Make horizontal line at 3.6% pension deduction
    ax.axhline(y=3.6, color="gray", linestyle="--", label="true ERP")
    ax.set_xticks(range(0, len(age_bins) - 1))
    ax.set_yticks(np.arange(0, 20, 2.5))

    # Create age range labels
    age_labels = [
        f"{age_bins[i]}-\n{age_bins[i + 1] - 1}" for i in range(len(age_bins) - 1)
    ]
    ax.set_xticklabels(age_labels, rotation=0)
    ax.legend(loc="upper left")
    ax.set_xlabel("Age")
    ax.set_ylabel("Penalty in %")
    ax.set_ylim([0, 20])
    fig.tight_layout()
    if save:
        plt.savefig(
            paths_dict["beliefs_plots"] + "erp_beliefs_by_age.png", bbox_inches="tight"
        )
    if show:
        plt.show()


def plot_informed_share_by_age(paths_dict, show=False, save=False):
    JET_COLOR_MAP, LINE_STYLES = set_colors()
    """
    Plot the share of informed individuals by age groups.
    Shows the proportion of each age group that is informed.
    """
    # Load and prepare the data
    df_soep_is = pd.read_csv(
        paths_dict["beliefs_data"] + "soep_is_clean.csv",
        dtype={"gebjahr": int},
    )

    # Select relevant columns
    relevant_columns = ["informed", "age"]
    data_informed = df_soep_is[~df_soep_is["informed"].isnull()][relevant_columns]

    # Create age bins and age groups
    age_bins = list(range(25, 71, 5))
    data_informed["age_group"] = create_age_groups(data_informed, age_bins=age_bins)

    # Calculate share of informed by age
    informed_by_age = (
        data_informed.groupby(["age_group"], observed=True)
        .agg({"informed": ["mean", "count"]})
        .round(4)
    )

    # Flatten column names
    informed_by_age.columns = ["informed_share", "count"]

    # Convert to percentage
    informed_by_age["informed_share_pct"] = informed_by_age["informed_share"] * 100

    # Create the plot
    fig, ax = plt.subplots()

    # Age labels
    age_labels = [
        f"{age_bins[i]}-\n{age_bins[i + 1] - 1}" for i in range(len(age_bins) - 1)
    ]

    # Create bar plot
    bars = ax.bar(
        range(len(informed_by_age)),
        informed_by_age["informed_share_pct"],
        color=JET_COLOR_MAP[0],
        alpha=0.7,
        linewidth=1,
    )

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=24,
        )

    # Customize axes
    ax.set_xlabel("Age")
    ax.set_ylabel("Share Informed (%)")
    ax.set_ylim([0, 50])
    ax.set_yticks(np.arange(0, 55, 5))
    ax.set_xticks(range(len(age_labels)))
    ax.set_xticklabels(age_labels)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0)

    fig.tight_layout()
    if save:
        plt.savefig(
            paths_dict["beliefs_plots"] + "informed_share_by_age.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()


def create_age_groups(data, age_bins):
    """Create age groups from continuous age variable."""
    return pd.cut(
        data["age"], bins=age_bins, labels=range(len(age_bins) - 1), right=False
    )


# Keep the figures cohort-based function names for backwards compatibility
def create_gebjahr_groups(data, age_bins):
    """Deprecated: Use create_age_groups instead."""
    return pd.cut(
        data["gebjahr"], bins=age_bins, labels=range(len(age_bins) - 1), right=False
    )
