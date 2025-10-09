import matplotlib.pyplot as plt
import numpy as np

from model_code.pension_system.early_retirement_paths import check_very_long_insured
from model_code.state_space.experience import (
    scale_experience_years,
)
from set_styles import get_figsize


def plot_ret_probs_for_state(model_solved, specs, path_dict):

    periods = np.arange(30, 40)

    # Low educated men not retired
    education = 0
    sex = 0
    lagged_choice = 3
    # Job offer and single
    job_offer = 1
    partner_state = 1
    assets = 22
    policy_state = 8

    n_obs = len(periods)
    int_array = np.ones(n_obs, dtype=int)
    float_array = np.ones(n_obs, dtype=float)

    states = {
        "period": periods,
        "policy_state": int_array * policy_state,
        "lagged_choice": int_array * lagged_choice,
        "education": int_array * education,
        "sex": int_array * sex,
        "job_offer": int_array * job_offer,
        "partner_state": int_array * partner_state,
        "assets_begin_of_period": float_array * assets,
    }

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=get_figsize(2, 2),
    )

    for id_exp, very_str in enumerate(["Very Long Insured", "Long Insured"]):

        exp_years = [45, 35][id_exp]

        exp_grid_float = scale_experience_years(
            experience_years=exp_years,
            period=periods,
            is_retired=False,
            model_specs=specs,
        )

        very_long_insured_bool = check_very_long_insured(
            retirement_age_difference=1,
            experience_years=exp_years,
            sex=sex,
            model_specs=specs,
        )
        if very_long_insured_bool:
            if very_str != "Very Long Insured":
                raise ValueError("Inconsistent very long insured status.")
        else:
            if very_str != "Long Insured":
                raise ValueError("Inconsistent very long insured status.")

        states["experience"] = float_array * exp_grid_float

        title_labels = [
            f"Good Health - {very_str}",
            f"Disability eligible - {very_str}",
        ]
        inform_labels = ["Not Informed", "Informed"]
        for id, health in enumerate([0, 2]):
            ax = axs[id, id_exp]
            for informed in [0, 1]:
                states["health"] = int_array * health
                states["informed"] = int_array * informed

                choice_probs = model_solved.choice_probabilities_for_states(
                    states=states
                )

                values = model_solved.choice_values_for_states(states=states)
                print(f"Values for {title_labels[id]}, informed {informed}")
                print(values[0, :])

                ax.plot(
                    specs["start_age"] + periods,
                    np.nan_to_num(choice_probs[:, 0], nan=2.0),
                    label=f"{inform_labels[informed]}",
                )
                ax.set_ylabel("Probability of retirement")

            ax.set_title(title_labels[id])

    axs[1, 0].set_xlabel("Age")
    axs[1, 1].set_xlabel("Age")

    axs[0, 0].legend()
    fig.savefig(path_dict["plots"] + f"retirement_probs_state_period_.png")

    plt.show()


def plot_work_probs_for_state(model_solved, specs, path_dict):

    # Vary periods, but fix SRA to 67 (policy state 8)
    period = 36
    policy_state = 4

    # Low educated men not retired
    education = 0
    sex = 0
    lagged_choice = 1
    # Job offer and single
    job_offer = 1
    health = 0
    exp_grid = np.arange(30, 50, 2, dtype=float)
    informed = 0

    n_obs = len(exp_grid)
    int_array = np.ones(n_obs, dtype=int)
    float_array = np.ones(n_obs, dtype=float)

    periods = int_array * period
    exp_grid_float = scale_experience_years(
        experience_years=exp_grid,
        period=periods,
        is_retired=False,
        model_specs=specs,
    )

    states = {
        "period": periods,
        "policy_state": int_array * policy_state,
        "lagged_choice": int_array * lagged_choice,
        "education": int_array * education,
        "sex": int_array * sex,
        "job_offer": int_array * job_offer,
        "experience": exp_grid_float,
        "health": int_array * health,
        "informed": int_array * informed,
    }

    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(10, 12),
    )

    for partner_state, partner_label in enumerate(specs["partner_labels"]):
        for assets in range(3, 18, 3):
            states["partner_state"] = int_array * partner_state
            states["assets_begin_of_period"] = float_array * assets

            choice_probs = model_solved.choice_probabilities_for_states(states=states)
            ax = axs[partner_state]
            ax.plot(
                exp_grid,
                np.nan_to_num(choice_probs[:, 3], nan=0.0),
                label=f"Assets {assets}",
            )
        ax.set_ylabel("Probability of working")
        ax.set_ylim([0, 1])

        ax.set_title(partner_label)
        ax.legend()

        ax.set_xlabel("Experience years")

    plt.show()
    fig.savefig(path_dict["plots"] + f"work_probs_state_period_{period}.png")
