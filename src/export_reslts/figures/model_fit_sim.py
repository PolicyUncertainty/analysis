import matplotlib.pyplot as plt
import pandas as pd
import yaml


def plot_average_wealth(paths):
    data_sim = pd.read_pickle(
        paths["intermediate_data"] + "sim_data_1_unc.pkl"
    ).reset_index()
    data_decision = pd.read_pickle(paths["intermediate_data"] + "decision_data.pkl")

    specs = yaml.safe_load(open(paths["specs"]))

    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_sim["age"] = data_sim["period"] + specs["start_age"]

    fig, ax = plt.subplots()
    data_sim.groupby("age")["wealth_at_beginning"].median().plot(
        label="Average " "simulated " "wealth by " "age", ax=ax
    )
    data_decision.groupby("age")["wealth"].median().plot(
        ax=ax, label="Observed wealth " "by " "age", ls="--"
    )
    ax.legend()


def plot_choice_shares_single(paths):
    data_sim = pd.read_pickle(
        paths["intermediate_data"] + "sim_data_1_unc.pkl"
    ).reset_index()
    data_decision = pd.read_pickle(paths["intermediate_data"] + "decision_data.pkl")

    specs = yaml.safe_load(open(paths["specs"]))

    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_sim["age"] = data_sim["period"] + specs["start_age"]

    choice_shares_sim = (
        data_sim.groupby(["age"])["choice"].value_counts(normalize=True).unstack()
    )
    choice_shares_obs = (
        data_decision.groupby(["age"])["choice"].value_counts(normalize=True).unstack()
    )

    fig, axes = plt.subplots(1, 3)
    for choice, ax in enumerate(axes):
        choice_shares_sim[choice].plot(ax=ax, label="Simulated")
        choice_shares_obs[choice].plot(ax=ax, label="Observed", ls="--")
        ax.set_title(f"Choice {choice}")
        ax.set_ylim([0, 1])
        ax.legend()


def plot_choice_shares(paths):
    data_sim = pd.read_pickle(
        paths["intermediate_data"] + "sim_data_1_unc.pkl"
    ).reset_index()
    data_decision = pd.read_pickle(paths["intermediate_data"] + "decision_data.pkl")

    specs = yaml.safe_load(open(paths["specs"]))

    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_sim["age"] = data_sim["period"] + specs["start_age"]

    data_sim.groupby(["age"])["choice"].value_counts(normalize=True).unstack().plot(
        title="Simulated choice shares by age", kind="bar", stacked=True
    )

    data_decision.groupby(["age"])["choice"].value_counts(
        normalize=True
    ).unstack().plot(title="Observed choice shares by age", kind="bar", stacked=True)
