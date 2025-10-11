import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def announcement_timing_lc_plot(path_dict, model_name):

    res_df_life_cycle = pd.read_csv(
        path_dict["sim_results"] + f"announcement_lc_{model_name}.csv", index_col=0
    )

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    axs = [ax1, ax2, ax3]

    subset_ages = np.arange(30, 81, 2)
    filtered_df = res_df_life_cycle.loc[subset_ages]

    announcement_ages = [35, 45, 55]
    for announcement_age in announcement_ages:
        savings_rate_diff = filtered_df[f"savings_rate_diff_{announcement_age}"]
        employment_rate_diff = filtered_df[f"employment_rate_diff_{announcement_age}"]
        retirement_rate_diff = filtered_df[f"retirement_rate_diff_{announcement_age}"]
        axs[0].plot(
            filtered_df.index,
            savings_rate_diff,
            label=f"SRA announcement at age {announcement_age}",
        )
        axs[1].plot(
            filtered_df.index,
            employment_rate_diff,
            label=f"SRA announcement age {announcement_age}",
        )
        axs[2].plot(
            filtered_df.index,
            retirement_rate_diff,
            label=f"SRA announcement age {announcement_age}",
        )

    axs[0].set_title("Savings rate difference")
    axs[1].set_title("Employment rate difference")
    axs[2].set_title("Retirement rate difference")
    # for ax in axs:
    #     ax.axhline(y=0, color="black")
    #     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.3f}"))
    for i in range(len(axs)):
        axs[i].legend()

    plot_folder = path_dict["simulation_plots"] + model_name + "/"
    fig1.savefig(plot_folder + "announcement_timing_savings_rate_diff.png")
    fig2.savefig(plot_folder + "announcement_timing_employment_rate_diff.png")
    fig3.savefig(plot_folder + "announcement_timing_retirement_rate_diff.png")
