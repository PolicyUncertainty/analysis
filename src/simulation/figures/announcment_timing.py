import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_life_cycle_plot(path_dict, model_name):

    res_df_life_cycle = pd.read_csv(
        path_dict["sim_results"] + f"announcement_lc_{model_name}.csv"
    )

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    subset_ages = np.arange(30, 81, 2)
    filtered_df = res_df_life_cycle.loc[subset_ages]

    annoucement_ages = [35, 45, 55]
    for announcement_age in annoucement_ages:
        savings_rate_diff = filtered_df[f"savings_rate_diff_{announcement_age}"]
        employment_rate_diff = filtered_df[f"employment_rate_diff_{announcement_age}"]
        retirement_rate_diff = filtered_df[f"retirement_rate_diff_{announcement_age}"]
        ax[0].plot(
            filtered_df.index,
            savings_rate_diff,
            label=f"SRA announcement at age {announcement_age}",
        )
        ax[1].plot(
            filtered_df.index,
            employment_rate_diff,
            label=f"SRA announcement age {announcement_age}",
        )
        ax[2].plot(
            filtered_df.index,
            retirement_rate_diff,
            label=f"SRA announcement age {announcement_age}",
        )

    ax[0].set_title("Savings rate difference")
    ax[1].set_title("Employment rate difference")
    ax[2].set_title("Retirement rate difference")
    for axis in ax:
        axis.axhline(y=0, color="black")
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.3f}"))
    plt.tight_layout()
    plt.legend()
    plt.show()
