import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import set_colors


def plot_sra_beliefs_by_age(paths_dict, show=False, save=False):
    JET_COLOR_MAP, LINE_STYLES = set_colors()
    df_soep_is = pd.read_csv(
        paths_dict["beliefs_data"] + "soep_is_clean.csv",
        dtype={"gebjahr": int},
    )

    df_soep_is.loc[:, "expected_stat_ret_age"] = (
        df_soep_is["pol_unc_stat_ret_age_67"] * 67
        + df_soep_is["pol_unc_stat_ret_age_68"] * 68
        + df_soep_is["pol_unc_stat_ret_age_69"] * 69
    )
    relevant_columns = [
        "pol_unc_stat_ret_age_67",
        "pol_unc_stat_ret_age_68",
        "pol_unc_stat_ret_age_69",
        "age",
    ]
    exp_ret_data = df_soep_is[~df_soep_is["expected_stat_ret_age"].isnull()][
        relevant_columns
    ]

    age_bins = list(range(25, 66, 5))

    exp_ret_data["age_group"] = create_age_groups(exp_ret_data, age_bins=age_bins)

    exp_ret_data_grouped = exp_ret_data.groupby(["age_group"], observed=True)
    exp_ret_data_mean = exp_ret_data_grouped[
        [
            "pol_unc_stat_ret_age_67",
            "pol_unc_stat_ret_age_68",
            "pol_unc_stat_ret_age_69",
        ]
    ].mean()

    fig, ax = plt.subplots()
    exp_ret_data_mean.plot(
        y=[
            "pol_unc_stat_ret_age_67",
            "pol_unc_stat_ret_age_68",
            "pol_unc_stat_ret_age_69",
        ],
        kind="bar",
        stacked=True,
        ax=ax,
        label=["67", "68", "69+"],
    )

    ax.set_xticks(range(0, len(age_bins) - 1))
    # Create age range labels
    age_labels = [
        f"{age_bins[i]}-\n{age_bins[i + 1] - 1}" for i in range(len(age_bins) - 1)
    ]
    ax.set_xticklabels(age_labels, rotation=0)
    # Make legebd central below the plot with three columns
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    ax.set_xlabel("Age")
    ax.set_ylabel("Attributed Percentage")
    ax.set_ylim([0, 100])
    ax.set_yticks(range(0, 100, 20))
    fig.tight_layout()
    if save:
        plt.savefig(
            paths_dict["beliefs_plots"] + "sra_beliefs_by_age.png", bbox_inches="tight"
        )
    if show:
        plt.show()


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


def plot_erp_violin_plots_by_age(paths_dict, show=False, save=False, censor_above=None):
    JET_COLOR_MAP, LINE_STYLES = set_colors()
    """
    Plot violin plots of ERP beliefs by age groups.
    Shows the distribution density of beliefs within each age group.

    Parameters:
    -----------
    paths_dict : dict
        Dictionary containing paths to data files
    show : bool, default False
        Whether to display the plot
    censor_above : float or None, default None
        If specified, values above this threshold will be censored (capped at this value)
    """
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

    # Apply censoring if specified
    if censor_above is not None:
        data_deduction = data_deduction.copy()
        data_deduction["belief_pens_deduct"] = data_deduction[
            "belief_pens_deduct"
        ].clip(upper=censor_above)

    age_bins = list(range(25, 71, 5))
    data_deduction["age_group"] = create_age_groups(data_deduction, age_bins=age_bins)

    # Create the violin plot
    fig, ax = plt.subplots()

    # Prepare data for violin plot
    violin_data = []
    age_labels = [
        f"{age_bins[i]}-\n{age_bins[i + 1] - 1}" for i in range(len(age_bins) - 1)
    ]

    # Extract data for each age group
    for group_id in range(len(age_bins) - 1):
        group_data = data_deduction[data_deduction["age_group"] == group_id][
            "belief_pens_deduct"
        ]
        if not group_data.empty:
            violin_data.append(group_data.values)
        else:
            violin_data.append([])

    # Create violin plot
    vp = ax.violinplot(
        violin_data,
        positions=range(1, len(age_labels) + 1),
        widths=0.8,
        showmeans=True,
        showmedians=True,
        bw_method=0.2,
    )

    # Customize violin plot appearance
    for pc in vp["bodies"]:
        pc.set_facecolor("white")
        pc.set_alpha(0.7)
        pc.set_edgecolor(JET_COLOR_MAP[0])
        pc.set_linewidth(2)

    # Customize the statistical lines
    vp["cmeans"].set_color(JET_COLOR_MAP[3])
    vp["cmeans"].set_linewidth(3)
    vp["cmedians"].set_color(JET_COLOR_MAP[1])
    vp["cmedians"].set_linewidth(3)
    vp["cbars"].set_color(JET_COLOR_MAP[7])
    vp["cbars"].set_linewidth(1)
    vp["cmins"].set_color(JET_COLOR_MAP[7])
    vp["cmins"].set_linewidth(1)
    vp["cmaxes"].set_color(JET_COLOR_MAP[7])
    vp["cmaxes"].set_linewidth(1)

    # Add horizontal line at 3.6% pension deduction (true ERP)
    ax.axhline(y=3.6, color="black", linestyle="--", linewidth=2, label="true ERP")

    # Add legend entries for mean and median lines
    import matplotlib.lines as mlines

    mean_line = mlines.Line2D(
        [], [], color=JET_COLOR_MAP[3], linewidth=2, label="mean of age group"
    )
    median_line = mlines.Line2D(
        [], [], color=JET_COLOR_MAP[1], linewidth=2, label="median of age group"
    )

    # Customize axes
    ax.set_xlabel("Age")
    ax.set_ylabel("Penalty in %")

    # Set y-axis limit based on censoring threshold
    if censor_above is not None:
        y_max = censor_above + 2  # 2 units above censoring threshold
    else:
        y_max = 50  # original default

    ax.set_ylim([0, y_max])
    ax.set_yticks(np.arange(0, (censor_above // 5 + 1) * 5, 5))
    ax.set_xticks(range(1, len(age_labels) + 1))
    ax.set_xticklabels(age_labels)

    # Create legend with all elements
    handles = [
        ax.lines[0],
        mean_line,
        median_line,
    ]  # true ERP line, mean line, median line
    labels = ["true ERP", "mean", "median"]

    ax.legend(
        handles=handles,
        labels=labels,
        loc="lower left",
        bbox_to_anchor=(0.11, 0.88),
        ncol=3,
        frameon=False,
    )

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0)

    fig.tight_layout()
    if save:
        plt.savefig(
            paths_dict["beliefs_plots"] + "erp_violin_plots_by_age.png",
            bbox_inches="tight",
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


# Keep the old cohort-based function names for backwards compatibility
def create_gebjahr_groups(data, age_bins):
    """Deprecated: Use create_age_groups instead."""
    return pd.cut(
        data["gebjahr"], bins=age_bins, labels=range(len(age_bins) - 1), right=False
    )
