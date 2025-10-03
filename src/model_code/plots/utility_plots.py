# Description: This file contains plotting functions for utility function results.
import matplotlib.pyplot as plt
import numpy as np

from model_code.utility.bequest_utility import utility_final_consume_all
from model_code.utility.utility_functions_add import consumption_scale, utility_func
from set_styles import get_figsize, set_colors


def plot_utility(path_dict, params, specs, show=False, save=False):
    """Plot utility function for different choices.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    params : dict
        Model parameters
    specs : dict
        Dictionary containing model specifications
    show : bool, default False
        Whether to display plots
    save : bool, default False
        Whether to save plots to disk
    """
    colors, _ = set_colors()
    consumption = np.linspace(5_000, 100_000, 1000) / specs["wealth_unit"]
    partner_state = np.array(1)
    education = 1
    period = 35

    choice_labels = specs["choice_labels"]
    fig, ax = plt.subplots()

    n_cons = len(consumption)
    utilities = np.zeros((n_cons, len(choice_labels)), dtype=float)
    for choice, choice_label in enumerate(choice_labels):
        for i, c in enumerate(consumption):
            utilities[i, choice] = utility_func(
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
            utilities[:, choice],
            consumption,
            label=choice_label,
            color=colors[choice % len(colors)],
        )

    ax.legend()
    ax.set_xlabel("Utility Men")
    ax.set_ylabel("Consumption")
    ax.set_title("Utility function (reversed axes)")

    plt.tight_layout()

    if save:
        fig.savefig(
            path_dict["model_plots"] + "utility_function.pdf", bbox_inches="tight"
        )
        fig.savefig(
            path_dict["model_plots"] + "utility_function.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bequest(path_dict, params, specs, show=False, save=False):
    """Plot bequest utility function.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    params : dict
        Model parameters
    specs : dict
        Dictionary containing model specifications
    show : bool, default False
        Whether to display plots
    save : bool, default False
        Whether to save plots to disk
    """
    colors, _ = set_colors()
    wealth = np.linspace(5_000, 100_000, 1000) / specs["wealth_unit"]

    choice_labels = specs["choice_labels"]
    fig, axs = plt.subplots(nrows=2, figsize=get_figsize(nrows=2, ncols=1))

    for sex in range(2):
        ax = axs[sex]
        for choice, choice_label in enumerate(choice_labels):
            bequests = np.zeros_like(wealth)
            for i, w in enumerate(wealth):
                bequests[i] = utility_final_consume_all(
                    wealth=w,
                    education=choice,  # Use choice as proxy for education
                    params=params,
                )
            ax.plot(
                wealth,
                bequests,
                label=choice_label,
                color=colors[choice % len(colors)],
            )
        ax.set_ylabel("Bequest Utility")
        ax.set_xlabel("Wealth")
        ax.set_title(f"{'Men' if sex == 0 else 'Women'}")
        if sex == 0:
            ax.legend()

    plt.tight_layout()

    if save:
        fig.savefig(
            path_dict["model_plots"] + "bequest_utility.pdf", bbox_inches="tight"
        )
        fig.savefig(
            path_dict["model_plots"] + "bequest_utility.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cons_scale(path_dict, specs, show=False, save=False):
    """Plot consumption scale by period.

    Parameters
    ----------
    path_dict : dict
        Dictionary containing paths to data and output directories
    specs : dict
        Dictionary containing model specifications
    show : bool, default False
        Whether to display plots
    save : bool, default False
        Whether to save plots to disk
    """
    colors, _ = set_colors()
    n_periods = specs["n_periods"]
    married_labels = ["Single", "Partnered"]
    edu_labels = specs["education_labels"]

    fig, axs = plt.subplots(ncols=2, figsize=get_figsize(ncols=2, nrows=1))

    for married_val, married_label in enumerate(married_labels):
        ax = axs[married_val]
        for edu_val, edu_label in enumerate(edu_labels):
            cons_scale = np.zeros(n_periods)
            for period in range(n_periods):
                cons_scale[period] = consumption_scale(
                    np.array(married_val), 0, edu_val, period, specs
                )
            ax.plot(cons_scale, label=edu_label, color=colors[edu_val])
            ax.set_title(married_label)
            ax.set_xlabel("Period")
            ax.legend()
            ax.set_ylim([1, 2])

    axs[0].set_ylabel("Consumption scale")
    fig.suptitle("Consumption scale by period")

    plt.tight_layout()

    if save:
        fig.savefig(
            path_dict["model_plots"] + "consumption_scale.pdf", bbox_inches="tight"
        )
        fig.savefig(
            path_dict["model_plots"] + "consumption_scale.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
