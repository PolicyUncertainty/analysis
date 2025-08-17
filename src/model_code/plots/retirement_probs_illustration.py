import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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
    n_exp = model_solved.value.shape[1]

    for exp_id in range(0, 8):
        period_fixed = np.ones_like(prototype_array) * period
        states = {
            "period": period_fixed,
            "lagged_choice": np.ones_like(prototype_array) * 3,
            "education": np.zeros_like(prototype_array),
            "sex": np.zeros_like(prototype_array),
            "informed": np.ones_like(prototype_array),
            "policy_state": np.ones_like(prototype_array) * 8,
            "job_offer": np.ones_like(prototype_array),
            "partner_state": np.zeros_like(prototype_array),
            "health": np.zeros_like(prototype_array),
            # "assets_begin_of_period": asset_grid
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

        # exp_grid_float = scale_experience_years(
        #     exp_years, period_fixed, specs["max_exp_diffs_per_period"]
        # )
        # states["experience"] = exp_grid_float

        endog_grid, value_grid, policy_grid = (
            model_solved.get_solution_for_discrete_state_choice(
                states=states, choices=np.ones_like(prototype_array) * 3
            )
        )

        ax.plot(
            endog_grid[0, exp_id, 1:],
            value_grid[0, exp_id, 1:],
            label=f"Exp years {exp_years}",
        )

    ax.legend()

    plt.show()


def plot_ret_probs_for_asset_level(model_solved, specs, path_dict):
    fig, ax = plt.subplots(figsize=(10, 6))

    # asset_grid = np.arange(1, 5)
    # n_obs = len(asset_grid)
    # exp_years_grid = np.linspace(10, 50, 5)
    # period = 40
    # n_exp = model_solved.value.shape[1]
    periods = np.arange(30, 40)
    n_obs = len(periods)
    prototype_array = np.arange(n_obs)

    for exp_years in range(0, 50, 10):
        states = {
            "period": periods,
            "lagged_choice": np.ones_like(prototype_array) * 3,
            "education": np.zeros_like(prototype_array),
            "sex": np.zeros_like(prototype_array),
            "informed": np.ones_like(prototype_array),
            "policy_state": np.ones_like(prototype_array) * 8,
            "job_offer": np.ones_like(prototype_array),
            "partner_state": np.zeros_like(prototype_array),
            "health": np.zeros_like(prototype_array),
            "assets_begin_of_period": np.ones_like(prototype_array) * 9,
        }

        exp_grid_float = scale_experience_years(
            experience_years=np.ones_like(prototype_array) * exp_years,
            period=periods,
            is_retired=states["lagged_choice"] == 0,
            model_specs=specs,
        )
        states["experience"] = exp_grid_float * np.ones_like(prototype_array)

        choice_probs = model_solved.choice_probabilities_for_states(states=states)

        ax.plot(
            periods,
            choice_probs[:, 0],
            label=f"Exp years {exp_years}",
        )

    ax.legend()

    plt.show()
