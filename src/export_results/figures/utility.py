# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# THIS IS THE LEGACY VERSION - DELETE SOON!
# NEW HOME: src/model_code/plots/utility_plots.py
# FUNCTIONS: plot_utility(), plot_bequest(), plot_cons_scale() migrated
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# %%
import matplotlib.pyplot as plt
import numpy as np

from model_code.utility.bequest_utility import utility_final_consume_all
from model_code.utility.utility_functions_cobb import consumption_scale, utility_func

# %%


def plot_utility(params, specs):
    consumption = np.linspace(5_000, 100_000, 1000) / specs["wealth_unit"]
    partner_state = np.array(1)
    education = 1
    period = 35

    choice_labels = specs["choice_labels"]
    fig, ax = plt.subplots()
    for choice, choice_label in enumerate(choice_labels):
        utilities = np.zeros_like(consumption)
        for i, c in enumerate(consumption):
            utilities[i] = utility_func(
                consumption=c,
                partner_state=partner_state,
                sex=0,
                health=1,
                education=education,
                period=period,
                choice=choice,
                params=params,
                model_specs=specs,
            )
        ax.plot(
            utilities,
            consumption,
            label=choice_label,
        )
    ax.legend()
    ax.set_xlabel("Utility")
    ax.set_ylabel("Consumption")
    ax.set_title("Utility function (reversed axes)")


def plot_bequest(params, specs):
    wealth = np.linspace(5_000, 100_000, 1000) / specs["wealth_unit"]

    choice_labels = specs["choice_labels"]
    fig, axs = plt.subplots(nrows=2)
    for sex in range(2):
        for choice, choice_label in enumerate(choice_labels):
            bequests = np.zeros_like(wealth)
            for i, w in enumerate(wealth):
                bequests[i] = utility_final_consume_all(
                    wealth=w,
                    sex=sex,
                    params=params,
                )
            axs[sex].plot(
                wealth,
                bequests,
                label=choice_label,
            )
    axs[0].legend()
    axs[0].set_ylabel("Bequest Utility")
    axs[0].set_xlabel("Consumption")
    axs[1].set_xlabel("Consumption")


def plot_cons_scale(specs):
    n_periods = specs["n_periods"]
    married_labels = ["Single", "Partnered"]
    edu_labels = specs["education_labels"]
    fig, axs = plt.subplots(ncols=2)
    for married_val, married_label in enumerate(married_labels):
        for edu_val, edu_label in enumerate(edu_labels):
            cons_scale = np.zeros(n_periods)
            for period in range(n_periods):
                cons_scale[period] = consumption_scale(
                    np.array(married_val), 0, edu_val, period, specs
                )
            axs[married_val].plot(cons_scale, label=edu_label)
            axs[married_val].set_title(married_label)
            axs[married_val].set_xlabel("Period")

            axs[married_val].legend()
            axs[married_val].set_ylim([1, 2])
    axs[0].set_ylabel("Consumption scale")
    fig.suptitle("Consumption scale by period")
