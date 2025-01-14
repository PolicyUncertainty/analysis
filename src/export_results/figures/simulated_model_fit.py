import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from estimation.struct_estimation.estimate_setup import load_and_prep_data
from model_code.specify_model import specify_model
from model_code.stochastic_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)


def plot_average_wealth(paths):
    data_sim = pd.read_pickle(
        paths["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
    ).reset_index()
    specs = yaml.safe_load(open(paths["specs"]))

    params = pickle.load(open(paths["est_results"] + "est_params_pete.pkl", "rb"))
    # Generate model_specs
    model, params = specify_model(
        path_dict=paths,
        update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
        policy_state_trans_func=expected_SRA_probs_estimation,
        params=params,
        load_model=True,
        model_type="solution",
    )
    data_decision, _ = load_and_prep_data(paths, params, model, drop_retirees=False)
    data_decision["age"] = data_decision["period"] + specs["start_age"]

    data_sim["age"] = data_sim["period"] + specs["start_age"]

    for sex in range(2):
        for edu in range(2):
            mask_sim = (data_sim["sex"] == sex) & (data_sim["education"] == edu)
            data_sim_sex = data_sim[mask_sim]
            mask_obs = (data_decision["sex"] == sex) & (
                data_decision["education"] == edu
            )
            data_decision_sex = data_decision[mask_obs]
            average_wealth_sim = data_sim_sex.groupby("age")["total_income"].median()
            average_wealth_obs = data_decision_sex.groupby("age")[
                "adjusted_wealth"
            ].median()

            ages = np.arange(specs["start_age"], specs["end_age"] + 1)
            average_wealth_obs_container = pd.DataFrame(
                index=ages, columns=["adjusted_wealth"], data=0, dtype=float
            )
            average_wealth_obs_container.update(average_wealth_obs)

            fig, ax = plt.subplots()
            ax.plot(ages, average_wealth_sim, label="Median simulated wealth by age")
            # ax.plot(
            #     ages,
            #     average_wealth_obs_container,
            #     label="Median observed wealth by age",
            #     ls="--",
            # )
            ax.legend()
            fig.savefig(
                paths["plots"] + "average_wealth.png", transparent=True, dpi=300
            )


def plot_choice_shares_single(paths):
    data_sim = pd.read_pickle(
        paths["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
    ).reset_index()
    data_decision = pd.read_pickle(
        paths["intermediate_data"] + "structural_estimation_sample.pkl"
    )

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
        paths["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
    ).reset_index()
    data_decision = pd.read_pickle(
        paths["intermediate_data"] + "structural_estimation_sample.pkl"
    )

    specs = yaml.safe_load(open(paths["specs"]))

    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_sim["age"] = data_sim["period"] + specs["start_age"]

    data_sim.groupby(["age"])["choice"].value_counts(normalize=True).unstack().plot(
        title="Simulated choice shares by age", kind="bar", stacked=True
    )

    data_decision.groupby(["age"])["choice"].value_counts(
        normalize=True
    ).unstack().plot(title="Observed choice shares by age", kind="bar", stacked=True)


def illustrate_simulated_data(paths):
    df = pd.read_pickle(
        paths["intermediate_data"] + "sim_data/data_subj_scale_1.pkl"
    ).reset_index()
    # import matplotlib.pyplot as plt
    #
    # # %%
    # # Plot choice shares by age
    # df.groupby(["age"]).choice.value_counts(normalize=True).unstack().plot(
    #     title="Choice shares by age"
    # )
    #
    # # %%fig_1 = (
    #
    # # plot average income by age and choice
    # df.groupby(["age", "choice"])["labor_income"].mean().unstack().plot(
    #     title="Average income by age and choice"
    # )
    # # plot average income by age and choice
    # df.groupby(["age", "choice"])["total_income"].mean().unstack().plot(
    #     title="Average total income by age and choice"
    # )
    # %%
    # plot average consumption by age and choice
    # df.groupby(["age", "choice"])["consumption"].mean().unstack().plot(
    #     title="Average consumption by age and choice"
    # )
    # %%
    fig, ax = plt.subplots()
    # plot average periodic savings by age and choice
    df.groupby("age")["savings_dec"].mean().plot(ax=ax, label="Average savings")
    ax.set_title("Average savings by age")
    ax.legend()
    fig.savefig(paths["plots"] + "average_savings.png", transparent=True, dpi=300)

    # # %%
    # # plot average utility by age and choice
    # df.groupby(["age", "choice"])["utility"].mean().unstack().plot(
    #     title="Average utility by age and choice"
    # )
    # # %%
    # # plot average wealth by age and choice
    # df.groupby(["age", "choice"])["wealth_at_beginning"].mean().unstack().plot(
    #     title="Average wealth by age and choice"
    # )
    # df.groupby(["age"])["wealth_at_beginning"].mean().plot(
    #     title="Average wealth by age"
    # )
