import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dcegm.interface import get_state_choice_index_per_state
from dcegm.likelihood import calc_choice_prob_for_state_choices
from dcegm.solve import get_solve_func_for_model
from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.policy_states_belief import update_specs_exp_ret_age_trans_mat
from model_code.specify_model import specify_model
from model_code.wealth_and_budget.savings_grid import create_savings_grid


def create_ll_from_paths(start_params_all, path_dict, load_model):
    # Specify the model
    model_collection, options_collection, params = specify_model(
        path_dict=path_dict,
        params=start_params_all,
        update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
        policy_state_trans_func=expected_SRA_probs_estimation,
        load_model=load_model,
    )
    model_structure_main = model_collection["model_main"]["model_structure"]

    # Load data
    data_decision = pd.read_pickle(
        path_dict["intermediate_data"] + "structural_estimation_sample.pkl"
    )
    data_decision = data_decision[data_decision["lagged_choice"] != 2]
    data_decision["wealth"] = data_decision["wealth"].clip(lower=1e-16)
    # Now transform for dcegm
    states_dict = {
        name: data_decision[name].values
        for name in model_structure_main["state_space_names"]
    }

    # Create a function that calculates the choice probabilities for the solution of
    # the main model
    observed_state_choice_indexes = get_state_choice_index_per_state(
        states=states_dict,
        map_state_choice_to_index=model_structure_main["map_state_choice_to_index"],
        state_space_names=model_structure_main["state_space_names"],
    )
    n_choices_main = options_collection["options_main"]["model_params"]["n_choices"]
    model_funcs_main = model_collection["model_main"]["model_funcs"]

    def choice_probs_for_main_solution(value_main_solved, endog_grid_main, params):
        return calc_choice_prob_for_state_choices(
            value_solved=value_main_solved,
            endog_grid_solved=endog_grid_main,
            params=params,
            states=states_dict,
            choices=data_decision["choice"].values,
            state_choice_indexes=observed_state_choice_indexes,
            oberseved_wealth=data_decision["choice"].values,
            choice_range=np.arange(n_choices_main, dtype=int),
            compute_utility=model_funcs_main["compute_utility"],
        )

    # Create savings grid
    savings_grid = create_savings_grid()

    solve_func_old_age = get_solve_func_for_model(
        model=model_collection["model_old_age"],
        exog_savings_grid=savings_grid,
        options=options_collection["options_old_age"],
    )

    solve_func_main = get_solve_func_for_model(
        model=model_collection["model_main"],
        exog_savings_grid=savings_grid,
        options=options_collection["options_main"],
    )

    def likelihood_func(params_in):
        params_updated = start_params_all.copy()
        params_updated.update(params_in)

        (value_old_age, policy_old_age, endog_grid_old_age) = solve_func_old_age(
            params_updated
        )

        params_updated["value_old_age"] = value_old_age
        params_updated["policy_old_age"] = policy_old_age
        params_updated["endog_grid_old_age"] = endog_grid_old_age

        value_main, policy_main, endog_grid_main = solve_func_main(params_updated)

        choice_probs = choice_probs_for_main_solution(
            value_main_solved=value_main,
            endog_grid_main=endog_grid_main,
            params=params_updated,
        ).clip(min=1e-10)
        likelihood_contributions = jnp.log(choice_probs)
        log_value = jnp.sum(-likelihood_contributions)
        return log_value, likelihood_contributions

    return jax.jit(likelihood_func)
