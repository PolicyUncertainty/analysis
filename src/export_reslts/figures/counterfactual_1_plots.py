import matplotlib.pyplot as plt
import numpy as np
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


def plot_values_by_age(paths_dict):
    data_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data_1_unc.pkl"
    ).reset_index()
    data_no_unc = pd.read_pickle(
        paths_dict["intermediate_data"] + "sim_data_1_no_unc.pkl"
    ).reset_index()

    est_params = pd.read_pickle(paths_dict["est_results"] + "est_params.pkl")

    specs = read_and_derive_specs(paths_dict["specs"])

    data_unc["age"] = data_unc["period"] + specs["start_age"]
    data_no_unc["age"] = data_no_unc["period"] + specs["start_age"]

    data_unc["real_taste_shock"] = np.nan
    data_unc.loc[data_unc["choice"] == 0, "real_taste_shock"] = data_unc.loc[
        data_unc["choice"] == 0, "taste_shock_0"
    ]
    data_unc.loc[data_unc["choice"] == 1, "real_taste_shock"] = data_unc.loc[
        data_unc["choice"] == 1, "taste_shock_1"
    ]
    data_unc.loc[data_unc["choice"] == 2, "real_taste_shock"] = data_unc.loc[
        data_unc["choice"] == 2, "taste_shock_2"
    ]
    data_no_unc["real_taste_shock"] = np.nan
    data_no_unc.loc[data_no_unc["choice"] == 0, "real_taste_shock"] = data_no_unc.loc[
        data_no_unc["choice"] == 0, "taste_shock_0"
    ]
    data_no_unc.loc[data_no_unc["choice"] == 1, "real_taste_shock"] = data_no_unc.loc[
        data_no_unc["choice"] == 1, "taste_shock_1"
    ]
    data_no_unc.loc[data_no_unc["choice"] == 2, "real_taste_shock"] = data_no_unc.loc[
        data_no_unc["choice"] == 2, "taste_shock_2"
    ]
    data_unc["real_util"] = data_unc["real_taste_shock"] + data_unc["utility"]
    data_no_unc["real_util"] = data_no_unc["real_taste_shock"] + data_no_unc["utility"]
    mean_real_util_unc = (
        data_unc.groupby("period")["real_util"].mean().sort_index().values
    )
    mean_real_util_no_unc = (
        data_no_unc.groupby("period")["real_util"].mean().sort_index().values
    )
    mean_disc_util_unc = mean_real_util_unc.copy()
    mean_disc_util_no_unc = mean_real_util_no_unc.copy()

    max_period = data_unc["period"].max()
    # reverse loop over range
    for i in range(max_period - 1, -1, -1):
        mean_disc_util_unc[i] += mean_disc_util_unc[i + 1] * est_params["beta"]
        mean_disc_util_no_unc[i] += mean_disc_util_no_unc[i + 1] * est_params["beta"]

    value_diff = mean_disc_util_unc - mean_disc_util_no_unc
    max_age = data_unc["age"].max()
    min_age = data_unc["age"].min()
    # plot average value by age
    fig, ax = plt.subplots()
    ax.plot(range(min_age, max_age + 1), mean_disc_util_unc, label="Uncertainty")
    ax.plot(range(min_age, max_age + 1), mean_disc_util_no_unc, label="No Uncertainty")
    # ax.plot(range(min_age, max_age  +1), value_diff, label="No Uncertainty")
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
