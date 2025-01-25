import matplotlib.pyplot as plt
import numpy as np
from export_results.figures.color_map import JET_COLOR_MAP


def plot_percentage_change(df, path_dict, cf_name, show_sra_diff=True):
    """Plot the change in baseline outcomes as a percentage of the baseline outcome."""

    reform_SRA = [67, 68, 69, 70]
    df = df[df["sra_at_63"].isin(reform_SRA)]
    fig, axs = plt.subplots(nrows=3, figsize=(12, 8))

    row_names = {
        "below_sixty_savings": "Changes in Savings",
        "working_hours": "Changes in Working Hours",
    }

    for i, var in enumerate(row_names.keys()):
        ax = axs[i]
        change = df[var] / df["base_" + var] - 1
        ax.plot(df["sra_at_63"], change * 100, label=row_names[var])

        # ax.plot(df["alpha"], df["cv"], label="Compensated Variation")
        ax.set_ylabel(f"Percentage change")
        ax.legend()
        ax.set_xticks(reform_SRA)

    ax = axs[2]
    # ax.yaxis.set_visible(False)

    # # Create a twin axis only for the third column (rightmost subplot)
    # ax_right = ax.twinx()
    # ax_right.set_ylabel("Change in years")

    ax.plot(df["sra_at_63"], df["ret_age"], label="Simulated Retirement Age")
    if show_sra_diff:
        ax.plot(
            df["sra_at_63"],
            df["sra_at_ret"],
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

    fig.savefig(path_dict["plots"] + f"{cf_name}_behavior.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df["sra_at_63"], df["cv"], label="Compensating Variation")
    ax.set_xlabel("SRA Reform")
    ax.set_ylabel("Compensating Variation")
    ax.set_xticks(reform_SRA)

    ax.legend()

    fig.savefig(path_dict["plots"] + f"{cf_name}_cv.png")
