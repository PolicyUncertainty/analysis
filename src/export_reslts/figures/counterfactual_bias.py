import matplotlib.pyplot as plt
import pandas as pd
from model_code.derive_specs import read_and_derive_specs


def plot_savings_over_age(paths_dict):
    data_baseline = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_baseline.pkl"
    ).reset_index()

    data_05_scale = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_05_scale.pkl"
    ).reset_index()

    data_2_scale = pd.read_pickle(
        paths_dict["intermediate_data"] + "data_2_scale.pkl"
    ).reset_index()

    specs = read_and_derive_specs(paths_dict["specs"])

    data_baseline["age"] = data_baseline["period"] + specs["start_age"]
    data_05_scale["age"] = data_05_scale["period"] + specs["start_age"]
    data_2_scale["age"] = data_2_scale["period"] + specs["start_age"]

    mean_savings = data_baseline.groupby("age")["savings"].mean()
    mean_savings_05_scale = data_05_scale.groupby("age")["savings"].mean()
    mean_savings_2_scale = data_2_scale.groupby("age")["savings"].mean()

    fig, ax = plt.subplots()
    ax.plot(mean_savings[:34], label="Baseline")
    ax.plot(mean_savings_05_scale[:34], label="SRA increase half")
    ax.plot(mean_savings_2_scale[:34], label="SRA increase double")
    ax.legend()
    ax.set_ylabel("Mean savings by age")
    ax.set_xlabel("Age")
    fig.tight_layout()
    fig.savefig(paths_dict["plots"] + "mean_savings_by_age.png")
