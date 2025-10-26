import pickle as pkl

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from first_step_estimation.estimation.family_estimation import (
    calc_trans_mat_vectorized,
    predicted_shares_for_sample,
)
from process_data.first_step_sample_scripts.create_partner_transition_sample import (
    create_partner_transition_sample,
)
from set_styles import get_figsize, set_colors, set_plot_defaults


def plot_partner_shares(
    paths_dict, specs, load_data=False, show=False, save=False, paper_plot=False
):
    """Plot predicted vs empirical partner state shares.

    Parameters
    ----------
    paths_dict : dict
        Dictionary containing paths to data and output directories
    specs : dict
        Dictionary containing model specifications
    load_data : bool, default False
        Whether to load data from disk
    show : bool, default False
        Whether to display plots
    save : bool, default False
        Whether to save plots to disk
    paper_plot : bool, default False
        Whether to create separate figures for paper
    """
    param_name_states = ["single", "working_age", "retirement"]

    est_data = create_partner_transition_sample(paths_dict, specs, load_data=load_data)
    est_data = est_data[
        (est_data["age"] >= specs["start_age"]) & (est_data["age"] <= specs["end_age"])
    ]

    all_ages = np.arange(specs["start_age"], specs["end_age"])
    old_ages = np.arange(specs["end_age_transition_estimation"] + 1, specs["end_age"])

    partner_state_vals = list(range(specs["n_partner_states"]))

    jet_color_map, _ = set_colors()
    set_plot_defaults()

    if paper_plot:
        figs = []
        axs_list = []
        for _ in range(6):  # 2 sexes × 3 partner states
            fig, ax = plt.subplots()
            figs.append(fig)
            axs_list.append(ax)
    else:
        fig, axs = plt.subplots(2, 3, figsize=get_figsize(2, 3))

    titles = []
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for i in range(3):
            sex_label_lower = sex_label.lower()
            state_label = param_name_states[i]
            titles.append(f"partner_lifecycle_{sex_label_lower}_{state_label}")
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            df = est_data[
                (est_data["sex"] == sex_var) & (est_data["education"] == edu_var)
            ].copy()

            # Calculate empirical shares
            empirical_counts = df.groupby("age")["partner_state"].value_counts()
            mulitindex = pd.MultiIndex.from_product(
                [all_ages, [0, 1, 2]], names=["age", "partner_state"]
            )
            empirical_counts = empirical_counts.reindex(mulitindex, fill_value=0)
            n_obs = empirical_counts.groupby("age").transform("sum")
            empirical_shares = empirical_counts / n_obs
            # We manipulate the empirical shares and assign to all ages above end_age_transition_estimation
            # and older the empirical single share of the end_age_transition_estimation. Marriage shares,
            # we set to zero and put the whole share to retirement.
            empirical_shares.loc[(old_ages, 0)] = empirical_shares.loc[
                (specs["end_age_transition_estimation"], 0)
            ]
            empirical_shares.loc[(old_ages, 1)] = 0.0
            empirical_shares.loc[(old_ages, 2)] = (
                1 - empirical_shares.loc[(specs["end_age_transition_estimation"], 0)]
            )

            params = pkl.load(
                open(
                    paths_dict["first_step_results"]
                    + f"result_{sex_label}_{edu_label}.pkl",
                    "rb",
                )
            )

            initial_shares = empirical_shares.loc[
                (all_ages[0], partner_state_vals)
            ].values

            sra_weights = df.groupby("age")["SRA"].value_counts(normalize=True)
            unique_sras = df["SRA"].unique()
            sra_weights_dict = {
                sra: sra_weights.xs(sra, level="SRA")
                .reindex(all_ages, fill_value=0)
                .values
                for sra in unique_sras
            }
            pred_shares = predicted_shares_for_sample(
                params, all_ages, initial_shares, sra_weights_dict
            )

            for i in range(3):
                if paper_plot:
                    plot_idx = sex_var * 3 + i
                    ax = axs_list[plot_idx]

                else:
                    ax = axs[sex_var, i]

                edu_label_lower = edu_label.lower()
                ax.plot(
                    all_ages,
                    empirical_shares.xs(i, level="partner_state"),
                    label=f"obs. {edu_label_lower}",
                    ls="--",
                    color=jet_color_map[edu_var],
                )
                ax.plot(
                    all_ages,
                    pred_shares[:, i],
                    label=f"pred. {edu_label_lower}",
                    color=jet_color_map[edu_var],
                )
                ax.legend(frameon=False)
                ax.set_ylim([0, 1])
                ax.set_xlabel("Age")
                ax.set_ylabel("Share")

                if not paper_plot:
                    ax.set_title(f"{sex_label}, {param_name_states[i].capitalize()}")

    if paper_plot:
        for fig, title in zip(figs, titles):
            fig.tight_layout()
            fig.savefig(
                paths_dict["first_step_plots"] + f"{title}.png",
                bbox_inches="tight",
                dpi=300,
            )
    else:
        fig.tight_layout()
        if save:
            fig.savefig(
                paths_dict["first_step_plots"] + "partner_lifecycle.pdf",
                bbox_inches="tight",
            )
            fig.savefig(
                paths_dict["first_step_plots"] + "partner_lifecycle.png",
                bbox_inches="tight",
                dpi=300,
            )

    if show:
        plt.show()
    else:
        if paper_plot:
            for fig in figs:
                plt.close(fig)
        else:
            plt.close(fig)


def plot_trans_probs(ages, sra, params, param_state_names):
    trans_probs = calc_trans_mat_vectorized(
        params=params,
        age=ages,
        sra=sra,
    )

    n_states = len(param_state_names)
    fig, axs = plt.subplots(n_states, n_states)
    for current_state, current_state_label in enumerate(param_state_names):
        axs[current_state, 0].set_ylabel(f"Prob. from {current_state_label}")
        for next_state, next_state_label in enumerate(param_state_names):
            axs[current_state, next_state].plot(
                ages, trans_probs[:, current_state, next_state]
            )

    for next_state, next_state_label in enumerate(param_state_names):
        axs[-1, next_state].set_xlabel(f"Age")
        axs[0, next_state].set_title(f"to {next_state_label}")
