import matplotlib.pyplot as plt
import pandas as pd
from model_code.derive_specs import read_and_derive_specs


def plot_full_time(paths_dict):
    data_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data_1_unc.pkl"
    ).reset_index()
    data_no_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data_1_no_unc.pkl"
    ).reset_index()

    specs = read_and_derive_specs(paths_dict["specs"])

    data_unc["age"] = data_unc["period"] + specs["start_age"]
    data_no_unc["age"] = data_no_unc["period"] + specs["start_age"]

    full_time_unc = (
        data_unc.groupby("age")["choice"].value_counts(normalize=True).reset_index()
    )
    full_time_unc = full_time_unc[full_time_unc["choice"] == 1]

    full_time_no_unc = (
        data_no_unc.groupby("age")["choice"].value_counts(normalize=True).reset_index()
    )
    full_time_no_unc = full_time_no_unc[full_time_no_unc["choice"] == 1]

    fig, ax = plt.subplots()
    ax.plot(full_time_unc["age"], full_time_unc["proportion"], label="Uncertainty")
    ax.plot(
        full_time_no_unc["age"], full_time_no_unc["proportion"], label="No Uncertainty"
    )
    ax.legend()
    ax.set_title("Proportion of full time workers by age")
    plt.show()


def plot_values_by_age(paths_dict):
    data_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data_1_unc.pkl"
    ).reset_index()
    data_no_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data_1_no_unc.pkl"
    ).reset_index()

    specs = read_and_derive_specs(paths_dict["specs"])

    data_unc["age"] = data_unc["period"] + specs["start_age"]
    data_no_unc["age"] = data_no_unc["period"] + specs["start_age"]

    # plot average value by age
    fig, ax = plt.subplots()
    ax.plot(data_unc.groupby("age")["value"].mean(), label="Uncertainty")
    ax.plot(data_no_unc.groupby("age")["value"].mean(), label="No Uncertainty")
    ax.set_title("Average value by age")
    ax.legend()


def plot_average_savings(paths_dict):
    data_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data_1_unc.pkl"
    ).reset_index()
    data_no_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data_1_no_unc.pkl"
    ).reset_index()

    specs = read_and_derive_specs(paths_dict["specs"])

    data_unc["age"] = data_unc["period"] + specs["start_age"]
    data_no_unc["age"] = data_no_unc["period"] + specs["start_age"]

    savings_unc = data_unc.groupby("age")["savings_dec"].mean()
    savings_no_unc = data_no_unc.groupby("age")["savings_dec"].mean()

    fig, ax = plt.subplots()
    ax.plot(savings_unc, label="Uncertainty")
    ax.plot(savings_no_unc, label="No Uncertainty")
    ax.set_title("Average savings by age")
    ax.legend()
