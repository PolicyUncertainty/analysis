import matplotlib.pyplot as plt
import numpy as np


def create_life_cycle_plot(res_df_life_cycle, sra_at_63):

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    subset_ages = np.arange(30, 81, 2)
    filtered_df = res_df_life_cycle.loc[subset_ages]

    colors = {
        67.0: "darkblue",
        68.0: "lightblue",
        69.0: "lightcoral",  # light red
        70.0: "darkred",
    }

    for i, sra in enumerate(sra_at_63):
        savings_diff = filtered_df[f"savings_rate_diff_{sra}"]
        labor_supply_diff = filtered_df[f"employment_rate_diff_{sra}"]
        retirement_diff = filtered_df[f"retirement_rate_diff_{sra}"]
        ax[0].plot(
            filtered_df.index, savings_diff, label=f"SRA at {sra}", color=colors[sra]
        )
        ax[1].plot(
            filtered_df.index, labor_supply_diff, label=f"SRA at {sra}", color=colors[sra]
        )
        ax[2].plot(
            filtered_df.index, retirement_diff, label=f"SRA at {sra}", color=colors[sra]
        )
    ax[0].set_title("Difference in savings rate")
    ax[1].set_title("Difference in employment rate")
    ax[2].set_title("Difference in retirement rate")
    for axis in ax:
        axis.axhline(y=0, color='black')
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.3f}'))
    plt.tight_layout()
    plt.legend()
    plt.show()