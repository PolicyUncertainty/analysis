import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_retirement_timing_data(paths, specs):
    struct_est_sample = pd.read_pickle(paths["struct_est_sample"])
    fresh_retirees = struct_est_sample[
        (struct_est_sample["choice"] == 0) & (struct_est_sample["lagged_choice"] != 0)
    ]
    n_fresh_retirees = fresh_retirees.shape[0]

    # Calculate actual retirement age vs SRA
    fresh_retirees["age"] = fresh_retirees["period"] + specs["start_age"]
    fresh_retirees["SRA"] = (
        specs["min_SRA"] + fresh_retirees["policy_state"] * specs["SRA_grid_size"]
    )
    fresh_retirees["actual_ret_age_vs_SRA"] = (
        fresh_retirees["age"] - fresh_retirees["SRA"]
    )
    counts = fresh_retirees["actual_ret_age_vs_SRA"].value_counts().sort_index()
    percentages = (counts / n_fresh_retirees) * 100

    # bar chart of actual retirement age vs SRA as percentage of total
    percentages.plot(kind="bar")
    plt.xlabel("Actual retirement age vs SRA")
    plt.ylabel("Percentage of Just-Retired Individuals")
    plt.title("Actual Retirement Age vs SRA")
