import pickle

import estimagic as em
import numpy as np
import pandas as pd
from dcegm.likelihood import calc_choice_probs_for_observed_states
from dcegm.likelihood import create_individual_likelihood_function_for_model
from dcegm.likelihood import create_observed_choice_indexes
from dcegm.solve import get_solve_func_for_model
from derive_specs import generate_specs_and_update_params
from model_code.budget_equation import create_savings_grid
from model_code.policy_states import expected_SRA_probs_estimation
from model_code.specify_model import specify_model


def create_likelihood(data_decision, project_paths, start_params_all, load_model):
    data_dict, model, options, savings_grid = prepare_estimation_model_and_data(
        data_decision, project_paths, start_params_all, load_model
    )

    # Create likelihood function
    individual_likelihood = create_individual_likelihood_function_for_model(
        model=model,
        options=options,
        observed_states=data_dict["states"],
        observed_wealth=data_dict["wealth"],
        observed_choices=data_dict["choices"],
        exog_savings_grid=savings_grid,
        params_all=start_params_all,
    )
    return individual_likelihood


def compute_model_fit(project_paths, start_params_all, load_model, load_solution):
    intermediate_data = project_paths["intermediate_data"]

    data_decision = pd.read_pickle(intermediate_data + "decision_data.pkl")

    if load_solution:
        choice_probs_observations = pickle.load(
            open(intermediate_data + "est_choice_probs.pkl", "rb")
        )
    else:
        data_dict, model, options, savings_grid = prepare_estimation_model_and_data(
            data_decision, project_paths, start_params_all, load_model
        )

        solve_func = get_solve_func_for_model(model, savings_grid, options)
        value, policy_left, policy_right, endog_grid = solve_func(start_params_all)
        observed_state_choice_indexes = create_observed_choice_indexes(
            data_dict["states"], model
        )
        choice_probs_observations = calc_choice_probs_for_observed_states(
            value_solved=value,
            endog_grid_solved=endog_grid,
            params=start_params_all,
            observed_states=data_dict["states"],
            state_choice_indexes=observed_state_choice_indexes,
            oberseved_wealth=data_dict["choices"],
            choice_range=np.arange(options["model_params"]["n_choices"], dtype=int),
            compute_utility=model["model_funcs"]["compute_utility"],
        )
        choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
        pickle.dump(
            choice_probs_observations,
            open(intermediate_data + "est_choice_probs.pkl", "wb"),
        )

    data_decision["choice_0"] = 0
    data_decision["choice_1"] = 0
    data_decision["choice_2"] = 1
    data_decision.loc[
        data_decision["lagged_choice"] != 2, "choice_0"
    ] = choice_probs_observations[:, 0]
    data_decision.loc[
        data_decision["lagged_choice"] != 2, "choice_1"
    ] = choice_probs_observations[:, 1]
    data_decision.loc[
        data_decision["lagged_choice"] != 2, "choice_2"
    ] = choice_probs_observations[:, 2]
    return data_decision


def prepare_estimation_model_and_data(
    data_decision, project_paths, start_params_all, load_model
):
    # Generate model_specs
    project_specs, _ = generate_specs_and_update_params(
        project_paths, start_params_all, load_data=True
    )

    # Generate dcegm model for project specs
    model, options = specify_model(
        project_specs=project_specs,
        model_data_path=project_paths["intermediate_data"],
        exog_trans_func=expected_SRA_probs_estimation,
        load_model=load_model,
    )
    print("Model specified.")

    # Prepare data for estimation with information from dcegm model. Retirees don't
    # have any choice and therefore no information
    data_for_estimation = data_decision[data_decision["lagged_choice"] != 2]

    data_dict = {}
    # Now transform for dcegm
    data_dict["states"] = {
        name: data_for_estimation[name].values for name in model["state_space_names"]
    }
    data_dict["wealth"] = data_for_estimation["wealth"].values
    data_dict["choices"] = data_for_estimation["choice"].values

    # Specifiy savings wealth grid
    savings_grid = create_savings_grid()
    return data_dict, model, options, savings_grid


def visualize_em_database(db_path):
    fig = em.criterion_plot(db_path)
    fig.show()

    fig = em.params_plot(db_path)
    fig.show()
