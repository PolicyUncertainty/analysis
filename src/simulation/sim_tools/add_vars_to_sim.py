import numpy as np
from estimation.struct_estimation.scripts.observed_model_fit import (
    choice_probs_for_choice_vals,
)


def calc_choice_probs_for_df(
    data, model, params, est_model, wealth_col, unobserved_state_specs
):
    discrete_state_names = model["model_structure"]["discrete_states_names"]
    states_dict = {
        state_name: data[state_name].values for state_name in discrete_state_names
    }
    states_dict["wealth"] = data[wealth_col].values

    options = model["options"]
    if len(options["exog_grids"]) == 2:
        second_continuous_state_name = options["second_continuous_state_name"]
        states_dict[second_continuous_state_name] = data[
            second_continuous_state_name
        ].values

    n_choices = len(options["state_space"]["choices"])
    for choice in range(n_choices):
        choice_vals = np.ones_like(data["choice"].values) * choice

        choice_probs_observations = choice_probs_for_choice_vals(
            choice_vals=choice_vals,
            states_dict=states_dict,
            model=model,
            unobserved_state_specs=unobserved_state_specs,
            params=params,
            est_model=est_model,
            use_probability_of_observed_states=False,
        )

        choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
        data[f"prob_{choice}"] = choice_probs_observations
    return data
