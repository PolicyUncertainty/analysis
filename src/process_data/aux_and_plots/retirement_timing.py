import matplotlib.pyplot as plt
import pandas as pd


def plot_retirement_timing_data(paths, specs):
    struct_est_sample = pd.read_pickle(paths["struct_est_sample"])
    df_fresh = struct_est_sample[
        (struct_est_sample["choice"] == 0) & (struct_est_sample["lagged_choice"] != 0)
    ]

    # Calculate actual retirement age vs SRA
    df_fresh["age"] = df_fresh["period"] + specs["start_age"]
    df_fresh["SRA"] = (
        specs["min_SRA"] + df_fresh["policy_state"] * specs["SRA_grid_size"]
    )
    df_fresh["actual_ret_age_vs_SRA"] = df_fresh["age"] - df_fresh["SRA"]

    # bar chart of actual retirement age vs SRA as percentage of total
    fig, axs = plt.subplots(2, 2)
    # Plot in first plot distance and in second the age
    for sex, sex_label in enumerate(specs["sex_labels"]):
        df_sex = df_fresh[df_fresh["sex"] == sex]
        counts_distance = df_sex["actual_ret_age_vs_SRA"].value_counts().sort_index()
        counts_age = df_sex["age"].value_counts().sort_index()

        axs[sex, 0].plot(counts_distance)
        axs[sex, 0].set_title(f"Actual Retirement Age vs SRA; {sex_label}")

        axs[sex, 1].plot(counts_age)
        axs[sex, 1].set_title(f"Retirement Age; {sex_label}")

        axs[sex, 1].plot()

    plt.show()
