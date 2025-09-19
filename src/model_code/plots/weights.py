import numpy as np
from dcegm.likelihood import (
    create_choice_prob_function,
)

from model_code.transform_data_from_model import (
    create_states_dict,
    load_scale_and_correct_data,
)
from model_code.unobserved_state_weighting import create_unobserved_state_specs


def plot_weights(model_class, params, specs, paths):

    df = load_scale_and_correct_data(paths, model_class)

    states_dict = create_states_dict(df=df, model_class=model_class)

    unobserved_state_specs = create_unobserved_state_specs(df)

    _, weight_func = create_choice_prob_function(
        model_structure=model_class.model_structure,
        model_specs=model_class.model_specs,
        model_funcs=model_class.model_funcs,
        model_config=model_class.model_config,
        observed_states=states_dict,
        observed_choices=df["choice"].values,
        unobserved_state_specs=unobserved_state_specs,
        use_probability_of_observed_states=False,
        return_weight_func=True,
    )
    (
        weights,
        observed_weights,
        possible_states,
        weighting_vars_for_possible_states,
    ) = weight_func(params)
    # First check that the likelihood internally generates the correct unobserved state
    # and weight combination
    for unobs_name in unobserved_state_specs["observed_bools_states"].keys():
        for poss_state_id in range(len(possible_states)):
            assert np.equal(
                possible_states[poss_state_id][unobs_name],
                weighting_vars_for_possible_states[poss_state_id][f"{unobs_name}_new"],
            ).all()

    for poss_state_id in range(len(possible_states)):
        name = ""
        for unobs_name in unobserved_state_specs["observed_bools_states"].keys():
            unobs_bool = ~unobserved_state_specs["observed_bools_states"][unobs_name]
            unobs_val = np.unique(
                possible_states[poss_state_id][unobs_name][unobs_bool]
            )
            assert len(unobs_val) == 1
            name += f"{unobs_name}_{unobs_val[0]}_"

        df[name + "weight"] = weights[:, poss_state_id]
    integrate_weights = weights.sum(axis=1) * observed_weights
    df["observed_weight"] = observed_weights
    # breakpoint()
