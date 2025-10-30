import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from beliefs.soep_is.belief_data_plots import create_gebjahr_groups
from beliefs.soep_is.belief_data_plots_by_age import create_age_groups
from set_styles import set_colors


def plot_erp_violin_plots_by_age(paths_dict, show=False, save=False, censor_above=None):
    JET_COLOR_MAP, LINE_STYLES = set_colors()
    """
    Plot violin plots of ERP beliefs by age groups.
    Shows the distribution density of beliefs within each age group.
    When censoring is applied, violins show censored data but means are calculated from uncensored data.

    Parameters:
    -----------
    paths_dict : dict
        Dictionary containing paths to data files
    show : bool, default False
        Whether to display the plot
    censor_above : float or None, default None
        If specified, values above this threshold will be censored (capped at this value)
        for the violin display, but means will be calculated from uncensored data
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
    ].copy()

    age_bins = list(range(25, 71, 5))
    data_deduction["age_group"] = create_age_groups(data_deduction, age_bins=age_bins)

    # Create censored column if specified
    if censor_above is not None:
        data_deduction["belief_pens_deduct_censored"] = data_deduction[
            "belief_pens_deduct"
        ].clip(upper=censor_above)
    else:
        data_deduction["belief_pens_deduct_censored"] = data_deduction[
            "belief_pens_deduct"
        ]

    # Create the violin plot
    fig, ax = plt.subplots()

    # Prepare data for violin plot (censored)
    violin_data = []
    age_labels = [
        f"{age_bins[i]}-\n{age_bins[i + 1] - 1}" for i in range(len(age_bins) - 1)
    ]

    # Extract data for each age group
    for group_id in range(len(age_bins) - 1):
        group_data = data_deduction[data_deduction["age_group"] == group_id][
            "belief_pens_deduct_censored"
        ]
        if not group_data.empty:
            violin_data.append(group_data.values)
        else:
            violin_data.append([])

    # Create violin plot (without showing means, we'll add them manually)
    vp = ax.violinplot(
        violin_data,
        positions=range(1, len(age_labels) + 1),
        widths=0.8,
        showmeans=False,  # Changed to False
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

    # Manually add means from uncensored data as horizontal lines
    for group_id in range(len(age_bins) - 1):
        group_data_uncensored = data_deduction[data_deduction["age_group"] == group_id][
            "belief_pens_deduct"
        ]
        if not group_data_uncensored.empty:
            mean_val = group_data_uncensored.mean()
            # Get the violin width for this position
            pos = group_id + 1
            # Draw horizontal line for mean
            ax.plot(
                [pos - 0.18, pos + 0.18],
                [mean_val, mean_val],
                color=JET_COLOR_MAP[3],
                linewidth=3,
                zorder=3,
            )

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
            dpi=100,
        )
    if show:
        plt.show()


def plot_erp_violin_plots_by_cohort(
    paths_dict, show=False, save=False, censor_above=None
):
    JET_COLOR_MAP, LINE_STYLES = set_colors()
    """
    Plot violin plots of ERP beliefs by birth cohort groups.
    Shows the distribution density of beliefs within each cohort.

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
        "gebjahr",
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

    age_bins = [-np.inf] + list(range(1957, 2001, 5))
    data_deduction["gebjahr_group"] = create_gebjahr_groups(
        data_deduction, age_bins=age_bins
    )

    # Create the violin plot
    fig, ax = plt.subplots()

    # Prepare data for violin plot
    violin_data = []
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

    # Extract data for each cohort group
    for group_id in range(len(age_bins) - 1):
        group_data = data_deduction[data_deduction["gebjahr_group"] == group_id][
            "belief_pens_deduct"
        ]
        if not group_data.empty:
            violin_data.append(group_data.values)
        else:
            violin_data.append([])

    # Create violin plot
    vp = ax.violinplot(
        violin_data,
        positions=range(1, len(cohort_labels) + 1),
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

    mean_line = mlines.Line2D([], [], color=JET_COLOR_MAP[3], linewidth=2, label="mean")
    median_line = mlines.Line2D(
        [], [], color=JET_COLOR_MAP[1], linewidth=2, label="median"
    )

    # Customize axes
    ax.set_xlabel("Birth Cohort")
    ax.set_ylabel("Penalty in %")

    # Set y-axis limit based on censoring threshold
    if censor_above is not None:
        y_max = censor_above + 2  # 2 units above censoring threshold
    else:
        y_max = 50  # original default

    ax.set_ylim([0, y_max])
    ax.set_yticks(np.arange(0, (censor_above // 5 + 1) * 5, 5))
    ax.set_xticks(range(1, len(cohort_labels) + 1))
    ax.set_xticklabels(cohort_labels)

    # Create legend with all elements
    handles = [
        ax.lines[0],
        mean_line,
        median_line,
    ]  # true ERP line, mean line, median line
    labels = ["true ERP", "mean", "median"]

    # make the legend in the top of the plot just above 20% y axis with thrree columns
    # lower the legend a bit
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
            paths_dict["beliefs_plots"] + "erp_violin_plots_by_cohort.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()
