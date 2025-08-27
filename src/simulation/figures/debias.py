import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def debias_lc_plot(path_dict, model_name):

    res_df_life_cycle = pd.read_csv(
        path_dict["sim_results"] + f"debias_lc_{model_name}.csv", index_col=0
    )

    subset_ages = np.arange(30, 81, 2)
    filtered_df = res_df_life_cycle.loc[subset_ages]

    colors = {
        # 67.0: "darkblue",
        # 68.0: "lightblue",
        69.0: "lightcoral",  # light red
        # 70.0: "darkred",
    }
    sra_at_63 = [
        # 67.0, 68.0,
        69.0,
        # 70.0
    ]

    # plot the baseline
    fig, ax = plt.subplots()
    savings_base = filtered_df[f"savings_rate_67.0_base"]
    ax.plot(
        filtered_df.index,
        savings_base * 100,
        label=f"SRA at 67 baseline",
        color="black",
    )
    ax.set_title("Savings Rate baseline")
    ax.set_ylabel("Savings Rate (%)")
    fig.savefig(
        path_dict["plots"] + f"debias_life_cycle_baseline_{model_name}.png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for i, sra in enumerate(sra_at_63):
        savings_diff = filtered_df[f"savings_rate_diff_{sra}"]
        labor_supply_diff = filtered_df[f"employment_rate_diff_{sra}"]
        retirement_diff = filtered_df[f"retirement_rate_diff_{sra}"]
        ax[0].plot(
            filtered_df.index,
            savings_diff * 100,
            label=f"SRA at {sra}",
            color=colors[sra],
        )
        ax[1].plot(
            filtered_df.index,
            labor_supply_diff * 100,
            label=f"SRA at {sra}",
            color=colors[sra],
        )
        ax[2].plot(
            filtered_df.index,
            retirement_diff * 100,
            label=f"SRA at {sra}",
            color=colors[sra],
        )
    ax[0].set_title("Difference in savings rate")
    ax[1].set_title("Difference in employment rate")
    ax[2].set_title("Difference in retirement rate")
    ax[0].set_ylabel("Percentage points difference")
    ax[1].set_ylabel("Percentage points difference")
    ax[2].set_ylabel("Percentage points difference")

    # # Set y ticks to be integers
    # ax[0].set_yticks(np.arange(-3, 3, 1))
    # # Dont show decimal points in y axis
    # ax[1].set_yticks(np.arange(-3, 3, 1))
    # ax[2].set_yticks(np.arange(-1, 1.5, 0.5))

    plt.legend()
    for axis in ax:
        axis.axhline(y=0, color="black")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(
        path_dict["plots"] + f"debias_life_cycle_{model_name}.png",
        bbox_inches="tight",
        transparent=True,
    )
