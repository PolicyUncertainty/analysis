import os
import pickle
import pickle as pkl
import time

import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
import yaml
from dcegm.asset_correction import adjust_observed_assets
from matplotlib import pyplot as plt

from model_code.specify_model import specify_model
from model_code.state_space.experience import scale_experience_years
from model_code.transform_data_from_model import load_scale_and_correct_data
from set_styles import get_figsize, set_colors


def plot_income(path_dict, specs, show=False, save=False):
    """Plot simulated vs observed income by partner state and education.

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

    model_class = specify_model(
        path_dict,
        specs,
        subj_unc=True,
        custom_resolution_age=None,
        sim_specs=None,
        load_model=True,
        debug_info=None,
    )

    data_decision = load_scale_and_correct_data(path_dict, model_class)
    data_decision = data_decision[data_decision["lagged_choice"] != 1]

    obs_net_total_income = (
        data_decision.groupby(["partner_state", "sex", "education", "period"])[
            "last_year_hh_net_income"
        ].median()
        / specs["wealth_unit"]
    )

    sim_net_total_income = data_decision.groupby(
        ["partner_state", "sex", "education", "period"]
    )["net_hh_income"].median()
    sim_labor_income = data_decision.groupby(
        ["partner_state", "sex", "education", "period"]
    )["gross_labor_income"].median()

    fig, axs = plt.subplots(
        ncols=len(specs["education_labels"]),
        nrows=len(specs["partner_labels"]),
        figsize=get_figsize(
            nrows=len(specs["partner_labels"]), ncols=len(specs["education_labels"])
        ),
    )

    for partner_state, partner_label in enumerate(specs["partner_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axs[partner_state, edu_var]
            for sex_var, sex_label in enumerate(specs["sex_labels"]):
                ax.plot(
                    sim_net_total_income.loc[
                        (partner_state, sex_var, edu_var, slice(None))
                    ],
                    color=colors[sex_var],
                    label=f"Sim {sex_label}",
                )
                ax.plot(
                    obs_net_total_income.loc[
                        (partner_state, sex_var, edu_var, slice(None))
                    ],
                    color=colors[sex_var],
                    ls="--",
                    label=f"Obs {sex_label}",
                )
                ax.plot(
                    sim_labor_income.loc[
                        (partner_state, sex_var, edu_var, slice(None))
                    ],
                    color=colors[sex_var],
                    ls=":",
                    label=f"Sim Labor {sex_label}",
                )

                # ax.plot(
                #     sim_gross_labor_income.loc[(partner_state, sex_var, edu_var, slice(None))],
                #     color=colors[sex_var],
                #     label=f"Sim {sex_label}",
                # )
                # ax.plot(
                #     obs_gross_labor_income.loc[(partner_state, sex_var, edu_var, slice(None))],
                #     color=colors[sex_var],
                #     ls="--",
                #     label=f"Obs {sex_label}",
                # )

            if edu_var == 0:
                axs[partner_state, edu_var].set_ylabel(
                    f"{partner_label}\nHH Net Income"
                )
            if partner_state == len(specs["partner_labels"]) - 1:
                axs[partner_state, edu_var].set_xlabel("Period")
            if partner_state == 0:
                axs[partner_state, edu_var].set_title(edu_label)

    # axs[0, 0].set_ylabel("HH Net Income")
    # axs[1, 0].set_ylabel("HH Gross Labor income")  # This line would be for the second subplot type
    axs[0, 0].legend()

    plt.tight_layout()

    if save:
        fig.savefig(
            path_dict["data_plots"] + "hh_income_comparison.pdf", bbox_inches="tight"
        )
        fig.savefig(
            path_dict["data_plots"] + "hh_income_comparison.png",
            bbox_inches="tight",
            dpi=300,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
