import matplotlib.pyplot as plt
import numpy as np


def create_life_cycle_plot(res_df_life_cycle, annoucement_ages):

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    subset_ages = np.arange(30, 81, 2)
    filtered_df = res_df_life_cycle.loc[subset_ages]

    for announcement_age in annoucement_ages:
        savings_rate_diff = filtered_df[f"savings_rate_diff_{announcement_age}"]
        employment_rate_diff = filtered_df[f"employment_rate_diff_{announcement_age}"]
        retirement_rate_diff = filtered_df[f"retirement_rate_diff_{announcement_age}"]
        ax[0].plot(filtered_df.index, savings_rate_diff, label=f"SRA announcement at age {announcement_age}")
        ax[1].plot(filtered_df.index, employment_rate_diff, label=f"SRA announcement age {announcement_age}")
        ax[2].plot(filtered_df.index, retirement_rate_diff, label=f"SRA announcement age {announcement_age}")

    ax[0].set_title("Savings rate difference")
    ax[1].set_title("Employment rate difference")
    ax[2].set_title("Retirement rate difference")
    for axis in ax:
        axis.axhline(y=0, color='black')
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.3f}'))
    plt.tight_layout()
    plt.legend()
    plt.show()