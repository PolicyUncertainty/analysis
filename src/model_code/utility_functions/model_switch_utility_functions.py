import jax.numpy as jnp
from dcegm.interface import policy_for_state_choice_vec
from dcegm.interface import value_for_state_choice_vec


def create_switch_utility_functions_dict(old_age_model):
    old_age_model_struct = old_age_model["model_structure"]
    map_state_choice_to_index = jnp.array(
        old_age_model_struct["map_state_choice_to_index"]
    )
    state_space_names = old_age_model_struct["state_space_names"]
    utility_function_old_age = old_age_model["model_funcs"]["compute_utility"]
    marginal_utility_function_old_age = old_age_model["model_funcs"][
        "compute_marginal_utility"
    ]

    def switch_utility(
        resources,
        retirement_age_id,
        education,
        policy_state,
        experience,
        options,
        params,
    ):
        deduction_state = options["old_age_state_index_mapping"][
            policy_state, retirement_age_id
        ]
        state_choice_vec = {
            "period": 0,
            "lagged_choice": 0,
            "education": education,
            "experience": experience,
            "deduction_state": deduction_state,
            "dummy_exog": 0,
            "choice": 0,
        }

        value = value_for_state_choice_vec(
            state_choice_vec=state_choice_vec,
            wealth=resources,
            map_state_choice_to_index=map_state_choice_to_index,
            state_space_names=state_space_names,
            endog_grid_solved=params["endog_grid_old_age"],
            value_solved=params["value_old_age"],
            compute_utility=utility_function_old_age,
            params=params,
        )
        return value

    def marginal_switch_utility(
        resources,
        retirement_age_id,
        education,
        policy_state,
        experience,
        options,
        params,
    ):
        deduction_state = options["old_age_state_index_mapping"][
            policy_state, retirement_age_id
        ]
        state_choice_vec = {
            "period": 0,
            "lagged_choice": 0,
            "education": education,
            "experience": experience,
            "deduction_state": deduction_state,
            "dummy_exog": 0,
            "choice": 0,
        }

        policy = policy_for_state_choice_vec(
            state_choice_vec=state_choice_vec,
            wealth=resources,
            map_state_choice_to_index=map_state_choice_to_index,
            state_space_names=state_space_names,
            endog_grid_solved=params["endog_grid_old_age"],
            policy_solved=params["policy_old_age"],
        )

        marg_util = marginal_utility_function_old_age(
            **state_choice_vec,
            consumption=policy,
            params=params,
        )

        return marg_util

    return {
        "utility": switch_utility,
        "marginal_utility": marginal_switch_utility,
    }
