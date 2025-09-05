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
from set_styles import set_colors
from model_code.specify_model import specify_model
from model_code.state_space.experience import scale_experience_years
from process_data.structural_sample_scripts.create_structural_est_sample import (
    CORE_TYPE_DICT,
)


def plot_income(path_dict, specs, show=False, save=False):
    """Plot simulated vs observed income by sex and education.
    
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
    # Load data
    data_decision = pd.read_csv(path_dict["struct_est_sample"])
    data_decision = data_decision.astype(CORE_TYPE_DICT)

    # old people
    data_decision = data_decision[data_decision["period"] < 50]

    model_class = specify_model(
        path_dict,
        specs,
        subj_unc=True,
        custom_resolution_age=None,
        sim_specs=None,
        load_model=True,
        debug_info=None,
    )

    # Transform experience
    max_init_exp = scale_experience_years(
        period=data_decision["period"].values,
        experience_years=data_decision["experience"].values,
        is_retired=data_decision["lagged_choice"].values == 0,
        model_specs=model_class.model_specs,
    )

    # We can adjust wealth outside, as it does not depend on estimated parameters
    # (only on interest rate)
    # Now transform for dcegm
    states_dict = {
        name: data_decision[name].values
        for name in model_class.model_structure["discrete_states_names"]
    }
    states_dict["experience"] = data_decision["experience"].values
    states_dict["assets_begin_of_period"] = (
        data_decision["wealth"].values / specs["wealth_unit"]
    )

    model_name = specs["model_name"]

    params = pickle.load(
        open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )

    assets_begin_of_period, aux = adjust_observed_assets(
        observed_states_dict=states_dict,
        params=params,
        model_class=model_class,
        aux_outs=True,
    )

    data_decision["income_sim"] = (
        assets_begin_of_period - states_dict["assets_begin_of_period"]
    )

    # Joint incomes
    data_decision["joint_labor_income"] = (
        (data_decision["monthly_wage"] + data_decision["monthly_wage_partner"]) * 12
    ) / specs["wealth_unit"]
    data_decision["gross_hh_income"] = aux["gross_hh_income"]

    data_decision = data_decision[data_decision["partner_state"] > 0]

    sim_inc = data_decision.groupby(["sex", "education", "period"])[
        "gross_hh_income"
    ].mean()
    sim_inc_net = data_decision.groupby(["sex", "education", "period"])[
        "income_sim"
    ].mean()

    obs_inc_net = (
        data_decision.groupby(["sex", "education", "period"])[
            "last_year_hh_net_income"
        ].mean()
        / specs["wealth_unit"]
    )
    obs_inc = data_decision.groupby(["sex", "education", "period"])[
        "joint_labor_income"
    ].mean()

    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            ax = axs[edu_var]
            ax.plot(
                sim_inc.loc[(sex_var, edu_var, slice(None))],
                color=colors[sex_var],
                label=f"Sim {sex_label}",
            )
            ax.plot(
                obs_inc.loc[(sex_var, edu_var, slice(None))],
                color=colors[sex_var],
                ls="--",
                label=f"Obs {sex_label}",
            )
            # ax.plot(obs_inc_net.loc[(sex_var, edu_var, slice(None))], color=JET_COLOR_MAP[sex_var], ls=":")
            # ax.plot(sim_inc_net.loc[(sex_var, edu_var, slice(None))], color=JET_COLOR_MAP[sex_var], ls="-.", label=sex_label)

            ax.set_title(edu_label)
            ax.set_xlabel("Period")
            ax.set_ylabel("Income")
            ax.legend()

    plt.tight_layout()
    
    if save:
        fig.savefig(path_dict["data_plots"] + "income_comparison.pdf", bbox_inches="tight")
        fig.savefig(path_dict["data_plots"] + "income_comparison.png", bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
    else:
        plt.close(fig)
