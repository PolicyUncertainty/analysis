import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_results.figures.color_map import JET_COLOR_MAP


def sra_increase_aggregate_plot(path_dict, model_name):
    """Plot the change in baseline outcomes as a percentage of the baseline outcome."""

    df_unc = pd.read_csv(
        path_dict["sim_results"] + f"sra_increase_aggregate_unc_{model_name}.csv",
        index_col=0,
    )
    df_no_unc = pd.read_csv(
        path_dict["sim_results"] + f"sra_increase_aggregate_no_unc_{model_name}.csv",
        index_col=0,
    )
    df_debias = pd.read_csv(
        path_dict["sim_results"] + f"sra_increase_aggregate_debias_{model_name}.csv",
        index_col=0,
    )

    # Transform df to also have 67 data
    for column in df_unc.columns.values:
        if column in "sra_at_63":
            continue
        elif column != "cv":
            if "base" in column:
                df_unc.loc[0, column] = df_unc.loc[1, column]
                df_no_unc.loc[0, column] = df_no_unc.loc[1, column]
                df_debias.loc[0, column] = df_debias.loc[1, column]
            else:
                column_name_without_cf = column[2:]
                df_unc.loc[0, column] = df_unc.loc[1, "base" + column_name_without_cf]
                df_no_unc.loc[0, column] = df_no_unc.loc[
                    1, "base" + column_name_without_cf
                ]
                df_debias.loc[0, column] = df_debias.loc[
                    1, "base" + column_name_without_cf
                ]
        else:
            # Assign to 0 itemn 1 from base
            df_unc.loc[0, column] = 0.0
            df_no_unc.loc[0, column] = 0.0
            df_debias.loc[0, column] = 0.0

    reform_SRA = [67, 68, 69, 70]
    df_unc = df_unc[df_unc["sra_at_63"].isin(reform_SRA)]
    df_no_unc = df_no_unc[df_no_unc["sra_at_63"].isin(reform_SRA)]
    df_debias = df_debias[df_debias["sra_at_63"].isin(reform_SRA)]
    fig, axs = plt.subplots(ncols=3, figsize=(16, 9))

    row_names = {
        "below_sixty_savings": "Perc. Change Savings",
        "working_hours": "Perc. Change Hours",
    }

    for i, var in enumerate(row_names.keys()):
        ax = axs[i]
        change_unc = df_unc["cf_" + var] / df_unc["base_" + var] - 1
        change_no_unc = df_no_unc["cf_" + var] / df_no_unc["base_" + var] - 1
        change_debias = df_debias["cf_" + var] / df_debias["base_" + var] - 1
        ax.plot(
            df_unc["sra_at_63"],
            change_unc * 100,
            label="With Uncertainty",
        )
        ax.plot(
            df_no_unc["sra_at_63"],
            change_no_unc * 100,
            label="Without Uncertainty",
        )
        # ax.plot(
        #     df_debias["sra_at_63"],
        #     change_debias * 100,
        #     label="No Uncertainty and Misinformation",
        # )

        # ax.plot(df["alpha"], df["cv"], label="Compensated Variation")
        ax.set_ylabel(f"{row_names[var]}")
        ax.legend()
        # ax.set_xticks(reform_SRA)
        # Hide xticks
        ax.set_xticks(reform_SRA)
        ax.set_xlabel("SRA Reform")

    ax = axs[2]
    # ax.yaxis.set_visible(False)

    # # Create a twin axis only for the third column (rightmost subplot)
    # ax_right = ax.twinx()
    # ax_right.set_ylabel("Change in years")

    # Change of retirement age
    change_unc = df_unc["cf_ret_age"] - df_unc["base_ret_age"]
    change_no_unc = df_no_unc["cf_ret_age"] - df_no_unc["base_ret_age"]
    change_debias = df_debias["cf_ret_age"] - df_debias["base_ret_age"]

    ax.plot(df_unc["sra_at_63"], change_unc, label="With Uncertainty")
    ax.plot(df_no_unc["sra_at_63"], change_no_unc, label="Without Uncertainty")
    # ax.plot(
    #     df_debias["sra_at_63"], change_debias, label="No Uncertainty and Misinformation"
    # )
    # change of SRA
    change_unc = df_unc["cf_sra_at_ret"] - df_unc["base_sra_at_ret"]
    ax.plot(
        df_unc["sra_at_63"],
        change_unc,
        color=JET_COLOR_MAP[0],
        ls="--",
        label="45 degree",
    )

    # Set xticks to be the reform SRA
    ax.set_xticks(reform_SRA)
    ax.set_xlabel("SRA Reform")
    ax.set_ylabel("Change Retirement Age")

    ax.legend()
    fig.align_ylabels(axs)

    fig.savefig(path_dict["plots"] + f"cf_increase_behavior.png")

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(
        df_unc["sra_at_63"], df_unc["cv"] * 100, label="Uncertainty and Misinformation"
    )
    ax.plot(
        df_no_unc["sra_at_63"],
        df_no_unc["cv"] * 100,
        label="No Uncertainty, Misinformation",
    )
    # ax.plot(
    #     df_debias["sra_at_63"],
    #     df_debias["cv"] * 100,
    #     label="No Uncertainty and Misinformation",
    # )
    ax.set_xlabel("SRA Reform")
    ax.set_ylabel("Compensating Variation")
    ax.set_xticks(reform_SRA)

    ax.legend()

    fig.savefig(path_dict["plots"] + f"cf_increase_cv.png")
