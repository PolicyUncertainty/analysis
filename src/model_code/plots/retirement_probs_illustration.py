import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model_code.pension_system.early_retirement_paths import check_very_long_insured
from model_code.state_space.experience import (
    construct_experience_years,
    scale_experience_years,
)


def plot_solution(model_solved, specs, path_dict):
    fig, ax = plt.subplots(figsize=(10, 6))

    # asset_grid = np.arange(1, 5)
    # n_obs = len(asset_grid)
    n_obs = 1
    prototype_array = np.arange(n_obs)
    # exp_years_grid = np.linspace(10, 50, 5)
    period = 34
    choice = 3
    lagged_choice = 1
    policy_state = 2
    job_offer = 1
    sex = 0
    informed = 1

    for exp_id in range(4, 8):
        states = {
            "period": np.ones_like(prototype_array) * period,
            "lagged_choice": np.ones_like(prototype_array) * lagged_choice,
            "education": np.zeros_like(prototype_array),
            "sex": np.ones_like(prototype_array) * sex,
            "informed": np.ones_like(prototype_array) * informed,
            "policy_state": np.ones_like(prototype_array) * policy_state,
            "job_offer": np.ones_like(prototype_array) * job_offer,
            "partner_state": np.zeros_like(prototype_array),
            "health": np.zeros_like(prototype_array),
        }
        exp_float = model_solved.model_config["continuous_states_info"][
            "second_continuous_grid"
        ][exp_id]
        exp_years = construct_experience_years(
            float_experience=exp_float,
            period=period,
            is_retired=states["lagged_choice"] == 0,
            model_specs=specs,
        )
        print(exp_float)

        endog_grid, value_grid, policy_grid = (
            model_solved.get_solution_for_discrete_state_choice(
                states=states, choices=np.ones_like(prototype_array) * choice
            )
        )

        ax.plot(
            endog_grid[0, exp_id, 1:],
            value_grid[0, exp_id, 1:],
            label=f"Exp years {exp_years}",
        )

    ax.legend()

    plt.show()


def plot_ret_probs_for_state(model_solved, specs, path_dict):

    policy_states = np.arange(specs["n_policy_states"] - 1, dtype=int)
    policy_state_values = specs["min_SRA"] + policy_states * specs["SRA_grid_size"]
    # Vary periods, but fix SRA to 67 (policy state 8)
    period = 37

    # Low educated men not retired
    education = 0
    sex = 0
    lagged_choice = 3
    # Job offer and single
    job_offer = 1
    partner_state = 0
    exp_years = 45
    assets = 10

    n_obs = len(policy_states)
    int_array = np.ones(n_obs, dtype=int)
    float_array = np.ones(n_obs, dtype=float)

    periods = int_array * period

    exp_grid_float = scale_experience_years(
        experience_years=exp_years,
        period=periods,
        is_retired=False,
        model_specs=specs,
    )

    states = {
        "period": periods,
        "policy_state": policy_states,
        "lagged_choice": int_array * lagged_choice,
        "education": int_array * education,
        "sex": int_array * sex,
        "job_offer": int_array * job_offer,
        "partner_state": int_array * partner_state,
        "assets_begin_of_period": float_array * assets,
        "experience": float_array * exp_grid_float,
    }

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 12),
    )

    for id_very, very_str in enumerate(["Very Long Insured", "Long Insured"]):

        title_labels = [
            f"Good Health - {very_str}",
            f"Disability eligible - {very_str}",
        ]
        inform_labels = ["Not Informed", "Informed"]
        for id, health in enumerate([0, 2]):
            ax = axs[id, id_very]
            for informed in [0, 1]:
                states["health"] = int_array * health
                states["informed"] = int_array * informed

                choice_probs = model_solved.choice_probabilities_for_states(
                    states=states
                )

                ax.plot(
                    specs["start_age"] + period - policy_state_values,
                    np.nan_to_num(choice_probs[:, 0], nan=0.0),
                    label=f"{inform_labels[informed]}",
                )
                ax.set_ylabel("Probability of retirement")

            ax.set_title(title_labels[id])

    axs[1, 0].set_xlabel("SRA difference")
    axs[1, 1].set_xlabel("SRA difference")

    axs[0, 0].legend()
    fig.savefig(path_dict["plots"] + f"retirement_probs_state_period_.png")

    plt.show()
