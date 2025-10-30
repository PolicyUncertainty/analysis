import pandas as pd
from matplotlib import pyplot as plt

from beliefs.soep_is.belief_data_plots import create_gebjahr_groups
from beliefs.soep_is.belief_data_plots_by_age import create_age_groups
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
            paths_dict["beliefs_plots"] + "sra_beliefs_by_age.png",
            bbox_inches="tight",
            dpi=100,
        )
    if show:
        plt.show()


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
