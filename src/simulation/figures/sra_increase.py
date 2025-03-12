import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_results.figures.color_map import JET_COLOR_MAP


def plot_aggregate_results(path_dict, model_name):
    """Plot the change in baseline outcomes as a percentage of the baseline outcome."""

    df = pd.read_csv(
        path_dict["sim_results"] + f"sra_increase_aggregate_{model_name}.csv",
        index_col=0,
    )

    # Transform df to also have 67 data
    columns = [col for col in df.columns if "base_" not in col]
    for column in columns:
        if column == "sra_at_63":
            continue
        elif column != "cv":
            # Assign to 0 itemn 1 from base
            df.loc[0, column] = df.loc[1, "base_" + column]
            df.loc[0, "base_" + column] = df.loc[1, "base_" + column]
        else:
            # Assign to 0 itemn 1 from base
            df.loc[0, column] = 0.0

    reform_SRA = [67, 68, 69, 70]
    df = df[df["sra_at_63"].isin(reform_SRA)]
    fig, axs = plt.subplots(nrows=3, figsize=(6, 8))

    row_names = {
        "below_sixty_savings": "Changes in Savings",
        "working_hours": "Changes in Working Hours",
    }

    for i, var in enumerate(row_names.keys()):
        ax = axs[i]
        change = df[var] / df["base_" + var] - 1
        ax.plot(df["sra_at_63"], change * 100, label=row_names[var])

        # ax.plot(df["alpha"], df["cv"], label="Compensated Variation")
        ax.set_ylabel(f"Percentage Change")
        ax.legend()
        # ax.set_xticks(reform_SRA)
        # Hide xticks
        ax.set_xticks([])

    ax = axs[2]
    # ax.yaxis.set_visible(False)

    # # Create a twin axis only for the third column (rightmost subplot)
    # ax_right = ax.twinx()
    # ax_right.set_ylabel("Change in years")

    # Change of retirement age
    change = df["ret_age"] - df["base_ret_age"]

    ax.plot(df["sra_at_63"], change, label="Simulated Retirement Age")
    # change of SRA
    change = df["sra_at_ret"] - df["base_sra_at_ret"]
    ax.plot(
        df["sra_at_63"],
        change,
        color=JET_COLOR_MAP[0],
        ls="--",
        label="45 degree",
    )

    # Set xticks to be the reform SRA
    ax.set_xticks(reform_SRA)
    ax.set_xlabel("SRA Reform")
    ax.set_ylabel("Retirement Age")

    ax.legend()
    fig.align_ylabels(axs)

    fig.savefig(path_dict["plots"] + f"cf2_behavior.png")

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(df["sra_at_63"], df["cv"] * 100, label="Compensating Variation")
    ax.set_xlabel("SRA Reform")
    ax.set_ylabel("Compensating Variation")
    ax.set_xticks(reform_SRA)

    ax.legend()

    fig.savefig(path_dict["plots"] + f"cf2_cv.png")
