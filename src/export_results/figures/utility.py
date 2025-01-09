# %%
import matplotlib.pyplot as plt
import numpy as np
from model_code.utility.utility_functions import consumption_scale
from model_code.utility.utility_functions import utility_func

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
                education=education,
                period=period,
                choice=choice,
                params=params,
                options=specs,
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
                    np.array(married_val), sex, edu_val, period, specs
                )
            axs[married_val].plot(cons_scale, label=edu_label)
            axs[married_val].set_title(married_label)
            axs[married_val].set_xlabel("Period")

            axs[married_val].legend()
            axs[married_val].set_ylim([1, 2])
    axs[0].set_ylabel("Consumption scale")
    fig.suptitle("Consumption scale by period")
