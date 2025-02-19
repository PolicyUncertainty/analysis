import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from export_results.figures.color_map import JET_COLOR_MAP


def plot_savings(path_dict, model_name):
    """Plot the change in baseline outcomes as a percentage of the baseline outcome."""

    # Load results
    df = pd.read_csv(path_dict["sim_results"] + f"counterfactual_1_{model_name}.csv")
    breakpoint()
    reform_SRA = [67, 68, 69, 70]
    df = df[df["sra_at_63"].isin(reform_SRA)]
    fig, ax = plt.subplots(figsize=(8, 8))

    change_in_savings = df["below_sixty_savings"] / df["base_below_sixty_savings"] - 1
    change_in_savings *= 100
    ax.plot(
        df["sra_at_63"],
        change_in_savings,
        label="Changes in Savings",
        color=JET_COLOR_MAP[0],
    )

    ax.set_xlabel("SRA at Resolution")
    ax.set_ylabel("Percentage Change")
    ax.legend(loc="upper left")
    ax.set_xticks(reform_SRA)
    fig.savefig(path_dict["plots"] + f"cf1_savings.png")

    #
    # row_names = {
    #     "below_sixty_savings": "Changes in Savings",
    #     "working_hours": "Changes in Working Hours",
    # }
    #
    # for i, var in enumerate(row_names.keys()):
    #     ax = axs[i]
    #     change = df[var] / df["base_" + var] - 1
    #     ax.plot(df["sra_at_63"], change * 100, label=row_names[var])
    #
    #     # ax.plot(df["alpha"], df["cv"], label="Compensated Variation")
    #     ax.set_ylabel(f"Percentage change")
    #     ax.legend()
    #     # ax.set_xticks(reform_SRA)
    #
    # ax = axs[2]
    # # ax.yaxis.set_visible(False)
    #
    # # # Create a twin axis only for the third column (rightmost subplot)
    # # ax_right = ax.twinx()
    # # ax_right.set_ylabel("Change in years")
    #
    # # Change of retirement age
    # change = df["ret_age"] - df["base_ret_age"]
    #
    # ax.plot(df["sra_at_63"], change, label="Simulated Retirement Age")
    # if show_sra_diff:
    #     # change of SRA
    #     change = df["sra_at_ret"] - df["base_sra_at_ret"]
    #     ax.plot(
    #         df["sra_at_63"],
    #         change,
    #         color=JET_COLOR_MAP[0],
    #         ls="--",
    #         label="45 degree",
    #     )
    #
    # # Set xticks to be the reform SRA
    # ax.set_xticks(reform_SRA)
    # ax.set_xlabel("SRA Reform")
    # ax.set_ylabel("Retirement Age")
    #
    # ax.legend()
    # fig.align_ylabels(axs)
    #
    # fig.savefig(path_dict["plots"] + f"{cf_name}_behavior.png")
    #
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.plot(df["sra_at_63"], df["cv"], label="Compensating Variation")
    # ax.set_xlabel("SRA Reform")
    # ax.set_ylabel("Compensating Variation")
    # ax.set_xticks(reform_SRA)
    #
    # ax.legend()
    #
    # fig.savefig(path_dict["plots"] + f"{cf_name}_cv.png")

#def plotplot_savings_heterogeneity(path_dict, model_name):
