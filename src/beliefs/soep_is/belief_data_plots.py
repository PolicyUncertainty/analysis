import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import set_colors


def plot_sra_beliefs_by_cohort(paths_dict, show=False, save=False):
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
        "gebjahr",
    ]
    exp_ret_data = df_soep_is[~df_soep_is["expected_stat_ret_age"].isnull()][
        relevant_columns
    ]
    age_bins = list(range(1957, 2001, 5))

    exp_ret_data["gebjahr_group"] = create_gebjahr_groups(
        exp_ret_data, age_bins=age_bins
    )

    exp_ret_data_grouped = exp_ret_data.groupby(["gebjahr_group"], observed=True)
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
    # Replace birth cohort integers by two lined strings first row given the start date if cohor and second row the end date of cohort
    ax.set_xticks(range(0, 8))
    # Make the strings above such that they span two lines on x axis
    ax.set_xticklabels(
        [
            "1957-\n1961 ",
            "1962-\n1966 ",
            "1967-\n1971 ",
            "1972-\n1976 ",
            "1977-\n1981 ",
            "1982-\n1986 ",
            "1987-\n1991 ",
            "1992-\n1996 ",
        ],
        rotation=0,
    )
    ax.legend(loc="lower left")
    ax.set_xlabel("Birth Cohort")
    ax.set_ylabel("Attributed Percentage")
    ax.set_ylim([0, 100])
    ax.set_yticks(range(0, 100, 20))
    fig.tight_layout()
    if save:
        plt.savefig(
            paths_dict["beliefs_plots"] + "sra_beliefs_by_cohort.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()


def plot_erp_beliefs_by_cohort(paths_dict, show=False, save=False):
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
        "gebjahr",
    ]
    data_deduction = df_soep_is[~df_soep_is["belief_pens_deduct"].isnull()][
        relevant_columns
    ]
    age_bins = age_bins = [-np.inf] + list(range(1957, 2001, 5))
    data_deduction["gebjahr_group"] = create_gebjahr_groups(
        data_deduction, age_bins=age_bins
    )
    ded_data_edu_grouped = data_deduction.groupby(["gebjahr_group"], observed=True)
    ded_data_edu_mean = ded_data_edu_grouped["belief_pens_deduct"].mean()
    ded_data_edu_sem = ded_data_edu_grouped["belief_pens_deduct"].sem()
    ded_data_edu_median = ded_data_edu_grouped["belief_pens_deduct"].median()
    # Plot the results
    fig, ax = plt.subplots()
    ded_data_edu_mean.plot(
        y="belief_pens_deduct",
        ax=ax,
        label="mean ERP belief",
    )
    ded_data_edu_median.plot(
        y="belief_pens_deduct",
        ax=ax,
        #     color="grey",
        ls="--",
        label="median ERP belief",
    )
    ax.errorbar(
        x=ded_data_edu_mean.index,
        y=ded_data_edu_mean,
        yerr=ded_data_edu_sem,
        fmt="o",
        color="black",
        ecolor="grey",
        capsize=5,
    )
    # Make horizontal line at 3.6% pension deduction
    ax.axhline(y=3.6, color="gray", linestyle="--", label="true ERP")
    ax.set_xticks(range(0, 9))
    # Make the strings above such that they span two lines on x axis
    ax.set_yticks(np.arange(0, 20, 2.5))
    # # Make the strings above such that they span two lines on x axis
    ax.set_xticklabels(
        [
            "1956 &\nbefore ",
            "1957-\n1961 ",
            "1962-\n1966 ",
            "1967-\n1971 ",
            "1972-\n1976 ",
            "1977-\n1981 ",
            "1982-\n1986 ",
            "1987-\n1991 ",
            "1992-\n1996 ",
        ],
        rotation=0,
    )
    ax.legend(loc="upper left")
    ax.set_xlabel("Birth Cohort")
    ax.set_ylabel("Penalty in %")
    ax.set_ylim([0, 20])
    fig.tight_layout()
    if save:
        plt.savefig(
            paths_dict["beliefs_plots"] + "erp_beliefs_by_cohort.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()


def create_gebjahr_groups(data, age_bins):
    return pd.cut(
        data["gebjahr"], bins=age_bins, labels=range(len(age_bins) - 1), right=False
    )


def plot_informed_share_by_cohort(paths_dict, show=False, save=False):
    JET_COLOR_MAP, LINE_STYLES = set_colors()
    """
    Plot the share of informed individuals by birth cohort groups.
    Shows the proportion of each cohort that is informed.
    """
    # Load and prepare the data
    df_soep_is = pd.read_csv(
        paths_dict["beliefs_data"] + "soep_is_clean.csv",
        dtype={"gebjahr": int},
    )

    # Select relevant columns
    relevant_columns = ["informed", "gebjahr"]
    data_informed = df_soep_is[~df_soep_is["informed"].isnull()][relevant_columns]

    # Create age bins and cohort groups
    age_bins = [-np.inf] + list(range(1957, 2001, 5))
    data_informed["gebjahr_group"] = create_gebjahr_groups(
        data_informed, age_bins=age_bins
    )

    # Calculate share of informed by cohort
    informed_by_cohort = (
        data_informed.groupby(["gebjahr_group"], observed=True)
        .agg({"informed": ["mean", "count"]})
        .round(4)
    )

    # Flatten column names
    informed_by_cohort.columns = ["informed_share", "count"]

    # Convert to percentage
    informed_by_cohort["informed_share_pct"] = (
        informed_by_cohort["informed_share"] * 100
    )

    # Create the plot
    fig, ax = plt.subplots()

    # Cohort labels
    cohort_labels = [
        "1956 &\nbefore",
        "1957-\n1961",
        "1962-\n1966",
        "1967-\n1971",
        "1972-\n1976",
        "1977-\n1981",
        "1982-\n1986",
        "1987-\n1991",
        "1992-\n1996",
    ]

    # Create bar plot
    bars = ax.bar(
        range(len(informed_by_cohort)),
        informed_by_cohort["informed_share_pct"],
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
    ax.set_xlabel("Birth Cohort")
    ax.set_ylabel("Share Informed (%)")
    ax.set_ylim([0, 50])
    ax.set_yticks(np.arange(0, 55, 5))
    ax.set_xticks(range(len(cohort_labels)))
    ax.set_xticklabels(cohort_labels)

    # Add grid for better readability
    # ax.grid(axis='y', color=JET_COLOR_MAP[8])
    # ax.set_axisbelow(True)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0)

    fig.tight_layout()
    if save:
        plt.savefig(
            paths_dict["beliefs_plots"] + "informed_share_by_cohort.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()
