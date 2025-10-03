import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt

from model_code.state_space.experience import construct_experience_years
from set_styles import get_figsize


def plot_ret_solution(model_solved, specs, path_dict):

    # asset_grid = np.arange(1, 5)
    # n_obs = len(asset_grid)
    # exp_years_grid = np.linspace(10, 50, 5)
    period_grid = np.arange(0, 10, 3)
    asset_grid = np.arange(10, 100, 10)
    n_obs = len(asset_grid)
    prototype_array = np.arange(n_obs)

    choice = 0
    choice_2 = 0
    lagged_choice = 0
    policy_state = 8
    job_offer = 1
    sex = 0
    informed = 1
    health = 2
    partner_state = 1

    n_plots = 6
    exp_grids = np.arange(n_plots)
    fig, axs = plt.subplots(
        nrows=int(n_plots / 3), ncols=3, figsize=get_figsize(int(n_plots / 3), 3)
    )
    axs = axs.flatten()

    for i, period in enumerate(period_grid):

        ax = axs[i]
        for exp_id in exp_grids:
            exp_float = model_solved.model_config["continuous_states_info"][
                "second_continuous_grid"
            ][exp_id]

            states = {
                "period": np.ones_like(prototype_array) * period,
                "lagged_choice": np.ones_like(prototype_array) * lagged_choice,
                "education": np.zeros_like(prototype_array),
                "sex": np.ones_like(prototype_array) * sex,
                "informed": np.ones_like(prototype_array) * informed,
                "policy_state": np.ones_like(prototype_array) * policy_state,
                "job_offer": np.ones_like(prototype_array) * job_offer,
                "partner_state": np.ones_like(prototype_array) * partner_state,
                "health": np.ones_like(prototype_array) * health,
                "assets_begin_of_period": asset_grid,
                "experience": np.ones_like(prototype_array) * exp_float,
            }
            exp_years = construct_experience_years(
                float_experience=exp_float,
                period=states["period"],
                is_retired=states["lagged_choice"] == 0,
                model_specs=specs,
            )

            policy, value = model_solved.policy_and_value_for_states_and_choices(
                states=states, choices=np.ones_like(prototype_array) * choice
            )
            ax.plot(
                asset_grid,
                policy,
                label=f"Exp format {np.round(exp_float,1)}))",
            )
            ax.legend()
        # ax.set_title(exp_years)

        # endog_grid, value_grid, policy_grid = (
        #     model_solved.get_solution_for_discrete_state_choice(
        #         states=states, choices=np.ones_like(prototype_array) * choice_2
        #     )
        # )
        #
        # ax.plot(
        #     endog_grid[0, exp_id, :],
        #     value_grid[0, exp_id, :],
        #     label=f"Exp years {exp_years} ret",
        # )

    # ax.legend()
    # plt.show()

    fig.savefig(path_dict["plots"] + f"ret_solution.png")


def plot_solution(model_solved, specs, path_dict):
    fig, ax = plt.subplots()

    # asset_grid = np.arange(1, 5)
    # n_obs = len(asset_grid)
    n_obs = 1
    prototype_array = np.arange(n_obs)
    # exp_years_grid = np.linspace(10, 50, 5)
    period = 35
    choice = 3
    choice_2 = 0
    lagged_choice = 3
    policy_state = 0
    job_offer = 1
    sex = 0
    informed = 1

    for exp_id in range(8, 9):
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

        endog_grid, value_grid, policy_grid = (
            model_solved.get_solution_for_discrete_state_choice(
                states=states, choices=np.ones_like(prototype_array) * choice
            )
        )

        ax.plot(
            endog_grid[0, exp_id, 26:30],
            value_grid[0, exp_id, 26:30],
            label=f"Exp years {exp_years}",
        )

        endog_grid, value_grid, policy_grid = (
            model_solved.get_solution_for_discrete_state_choice(
                states=states, choices=np.ones_like(prototype_array) * choice_2
            )
        )

        ax.plot(
            endog_grid[0, exp_id, :],
            value_grid[0, exp_id, :],
            label=f"Exp years {exp_years} ret",
        )

    ax.legend()

    fig.savefig(path_dict["plots"] + f"solution_value_func_period_{period}.png")
