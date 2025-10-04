import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from set_styles import get_figsize, set_colors

JET_COLOR_MAP, LINE_STYLES = set_colors()


def load_het_results(path_dict, het_names, model_name):
    """Load all gender-specific result dataframes"""
    results = {}

    for scenario in ["unc", "no_unc"]:  # , "debias"]:
        results[scenario] = {}
        for gender in het_names:
            filepath = (
                path_dict["sim_results"]
                + f"sra_increase_aggregate_{scenario}_{gender}_{model_name}.csv"
            )
            results[scenario][gender] = pd.read_csv(filepath, index_col=0)

    return results


def prepare_baseline_data(df):
    """Transform df to also have baseline (67) data"""
    df_prepared = df.copy()

    for column in df_prepared.columns.values:
        if column == "sra_at_63":
            continue
        elif column != "cv":
            if "base" in column:
                df_prepared.loc[0, column] = df_prepared.loc[1, column]
            else:
                column_name_without_cf = column[2:]
                df_prepared.loc[0, column] = df_prepared.loc[
                    1, "base" + column_name_without_cf
                ]
        else:
            # CV is 0 for baseline
            df_prepared.loc[0, column] = 0.0

    return df_prepared


def plot_behavioral_changes(ax, df_unc, df_no_unc, var, ylabel, title):
    """Plot behavioral changes for a specific variable"""
    change_unc = df_unc["cf_" + var] / df_unc["base_" + var] - 1
    change_no_unc = df_no_unc["cf_" + var] / df_no_unc["base_" + var] - 1

    ax.plot(df_unc["sra_at_63"], change_unc * 100, label="With Uncertainty")
    ax.plot(df_no_unc["sra_at_63"], change_no_unc * 100, label="Without Uncertainty")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def plot_retirement_age_changes(ax, df_unc, df_no_unc, title, show_legend=False):
    """Plot retirement age changes"""
    # Change of retirement age
    change_ret_unc = df_unc["cf_ret_age"] - df_unc["base_ret_age"]
    change_ret_no_unc = df_no_unc["cf_ret_age"] - df_no_unc["base_ret_age"]

    ax.plot(df_unc["sra_at_63"], change_ret_unc, label="With Uncertainty")
    ax.plot(df_no_unc["sra_at_63"], change_ret_no_unc, label="Without Uncertainty")

    # Change of SRA (45 degree line reference)
    change_sra_unc = df_unc["cf_sra_at_ret"] - df_unc["base_sra_at_ret"]
    ax.plot(
        df_unc["sra_at_63"],
        change_sra_unc,
        color=JET_COLOR_MAP[0],
        ls="--",
        label="45 degree",
    )

    ax.set_ylabel("Change Retirement Age")
    ax.set_title(title)

    if show_legend:
        ax.legend()

    return ax


def sra_increase_aggregate_plot_by_het(path_dict, het_names, fig_name, model_name):
    """Plot the change in baseline outcomes as a percentage of the baseline outcome by gender."""

    het_names = ["overall"] + het_names
    # Load all results
    results = load_het_results(path_dict, het_names, model_name)

    plot_folder = path_dict["simulation_plots"] + model_name + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Prepare data for all scenarios and genders
    prepared_data = {}
    for scenario in ["unc", "no_unc"]:
        prepared_data[scenario] = {}
        for het_name in het_names:
            df = results[scenario][het_name]
            prepared_data[scenario][het_name] = prepare_baseline_data(df)

    # Filter for reform SRA values
    reform_SRA = [67, 68, 69, 70]
    for scenario in prepared_data:
        for het_name in prepared_data[scenario]:
            df = prepared_data[scenario][het_name]
            prepared_data[scenario][het_name] = df[df["sra_at_63"].isin(reform_SRA)]

    # Create the main behavioral changes plot (3 rows x 3 columns)
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=get_figsize(ncols=3, nrows=4))

    # Variables to plot
    variables = {
        "below_sixty_savings": "Perc. Change Savings",
        "working_hours": "Perc. Change Hours",
        "working_hours_below_63": "Perc. Change Hours < 63",
    }

    column_titles = ["Savings", "Working Hours", "Working Hours < 63", "Retirement Age"]

    # Plot behavioral changes for each gender (rows) and variable (columns)
    for row, het_name in enumerate(het_names):
        df_unc = prepared_data["unc"][het_name]
        df_no_unc = prepared_data["no_unc"][het_name]

        # Plot savings and working hours
        for col, (var, ylabel) in enumerate(variables.items()):
            ax = axs[row, col]
            plot_behavioral_changes(
                ax,
                df_unc,
                df_no_unc,
                var,
                ylabel,
                f"{column_titles[col]}" if row == 0 else "",
            )

            # Set x-axis properties
            ax.set_xticks(reform_SRA)
            if row == 2:  # Bottom row
                ax.set_xlabel("SRA Reform")

            # Add row labels on the left
            if col == 0:
                ax.set_ylabel(f"{het_name}\n{ylabel}")

        # Plot retirement age changes in third column
        ax = axs[row, 3]
        show_legend = row == 1  # Show legend only for middle row
        plot_retirement_age_changes(
            ax,
            df_unc,
            df_no_unc,
            f"{column_titles[3]}" if row == 0 else "",
            show_legend=show_legend,
        )

        ax.set_xticks(reform_SRA)
        if row == 2:  # Bottom row
            ax.set_xlabel("SRA Reform")

        if col == 0:
            ax.set_ylabel(f"{het_name}\nChange Retirement Age")

    # Add legend below the middle row plots
    axs[1, 1].legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=True
    )

    plt.tight_layout()
    fig.savefig(
        plot_folder + f"cf_increase_behavior_{fig_name}.png",
        transparent=True,
        bbox_inches="tight",
    )
    fig.savefig(
        plot_folder + f"cf_increase_behavior_{fig_name}.pdf", bbox_inches="tight"
    )

    # # Create compensating variation plot by gender
    # fig_cv, axs_cv = plt.subplots(nrows=3, ncols=1, figsize=get_figsize(ncols=1, nrows=3))

    # for row, gender in enumerate(gender_order):
    #     ax = axs_cv[row]
    #     df_unc = prepared_data["unc"][gender]
    #     df_no_unc = prepared_data["no_unc"][gender]

    #     ax.plot(df_unc["sra_at_63"], df_unc["cv"] * 100,
    #             label="Uncertainty and Misinformation")
    #     ax.plot(df_no_unc["sra_at_63"], df_no_unc["cv"] * 100,
    #             label="No Uncertainty, Misinformation")

    #     ax.set_ylabel(f"{row_titles[row]}\nCompensating Variation")
    #     ax.set_xticks(reform_SRA)

    #     if row == 2:  # Bottom row
    #         ax.set_xlabel("SRA Reform")

    #     if row == 0:  # Top row
    #         ax.set_title("Compensating Variation by Gender")

    #     if row == 1:  # Middle row
    #         ax.legend()

    # plt.tight_layout()
    # fig_cv.savefig(path_dict["plots"] + f"cf_increase_cv_by_gender.png",
    #                bbox_inches='tight')

    return fig


def sra_increase_aggregate_plot(path_dict, model_name):
    """
    Wrapper function to maintain backward compatibility with original function name.
    Creates both the original overall plots and the new gender-specific plots.
    """

    plot_folder = path_dict["simulation_plots"] + model_name + "/"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Also create the original overall plots for backward compatibility
    results = load_het_results(path_dict, het_names=["overall"], model_name=model_name)

    # Use overall results for original plots
    df_unc = prepare_baseline_data(results["unc"]["overall"])
    df_no_unc = prepare_baseline_data(results["no_unc"]["overall"])

    reform_SRA = [67, 68, 69, 70]
    df_unc = df_unc[df_unc["sra_at_63"].isin(reform_SRA)]
    df_no_unc = df_no_unc[df_no_unc["sra_at_63"].isin(reform_SRA)]

    # Original behavior plot
    fig, axs = plt.subplots(ncols=3, figsize=get_figsize(ncols=3, nrows=1))

    variables = {
        "below_sixty_savings": "Perc. Change Savings",
        "working_hours": "Perc. Change Hours",
    }
    labels = ["Savings", "Working Hours"]

    for i, (var, ylabel) in enumerate(variables.items()):
        ax = axs[i]
        plot_behavioral_changes(ax, df_unc, df_no_unc, var, ylabel, labels[i])
        ax.set_xticks(reform_SRA)
        ax.set_xlabel("SRA Reform")

    # Retirement age plot
    plot_retirement_age_changes(
        axs[2], df_unc, df_no_unc, "Retirement Age", show_legend=True
    )
    axs[2].set_xticks(reform_SRA)
    axs[2].set_xlabel("SRA Reform")

    # Add legend
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

    fig.align_ylabels(axs)
    fig.savefig(plot_folder + f"cf_increase_behavior.png", transparent=True)
    fig.savefig(plot_folder + f"cf_increase_behavior.pdf")

    # # Original CV plot
    # fig_cv_orig, ax_cv_orig = plt.subplots()
    # ax_cv_orig.plot(df_unc["sra_at_63"], df_unc["cv"] * 100,
    #                 label="Uncertainty and Misinformation")
    # ax_cv_orig.plot(df_no_unc["sra_at_63"], df_no_unc["cv"] * 100,
    #                 label="No Uncertainty, Misinformation")
    # ax_cv_orig.set_xlabel("SRA Reform")
    # ax_cv_orig.set_ylabel("Compensating Variation")
    # ax_cv_orig.set_xticks(reform_SRA)
    # ax_cv_orig.legend()
    # fig_cv_orig.savefig(path_dict["plots"] + f"cf_increase_cv.png")

    return fig
