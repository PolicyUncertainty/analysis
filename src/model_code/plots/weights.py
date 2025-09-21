import numpy as np
from dcegm.likelihood import (
    create_choice_prob_function,
)

from estimation.struct_estimation.scripts.observed_model_fit import create_df_with_probs
from model_code.transform_data_from_model import (
    create_states_dict,
)
from model_code.unobserved_state_weighting import create_unobserved_state_specs


def plot_weights(model_class, params, specs, paths):

    df = create_df_with_probs(
        path_dict=paths,
        params=params,
        model_name=specs["model_name"],
        load_sol_model=True,
        load_solution=True,
        unobs_choice_probs=True,
    )

    states_dict = create_states_dict(df=df, model_class=model_class)

    unobserved_state_specs = create_unobserved_state_specs(df)

    _, debug_func = create_choice_prob_function(
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
    ) = debug_func(params)
    # First check that the likelihood internally generates the correct unobserved state
    # and weight combination
    for unobs_name in unobserved_state_specs["observed_bools_states"].keys():
        for poss_state_id in range(len(possible_states)):
            assert np.equal(
                possible_states[poss_state_id][unobs_name],
                weighting_vars_for_possible_states[poss_state_id][f"{unobs_name}_new"],
            ).all()

    df["weighted_choice_0"] = 0.0
    for poss_state_id in range(len(possible_states)):
        name = ""
        for unobs_name in ["job_offer", "health", "informed"]:
            unobs_bool = ~unobserved_state_specs["observed_bools_states"][unobs_name]
            unobs_vals = np.unique(
                possible_states[poss_state_id][unobs_name][unobs_bool]
            )
            # First the case there are no unobserved states. Assign 0
            if len(unobs_vals) == 0:
                unobs_val = 0
            elif len(unobs_vals) == 1:
                unobs_val = unobs_vals[0]
            else:
                raise ValueError(
                    f"More than one unobserved state value for {unobs_name} and poss_state_id {poss_state_id}"
                )

            name += f"{unobs_name}_{unobs_val}_"

        df["weighted_choice_0"] += df[f"{name}choice_0"] * weights[:, poss_state_id]

        df[name + "weight"] = weights[:, poss_state_id]
    integrate_weights = weights.sum(axis=1) * observed_weights
    df["observed_weight"] = observed_weights
    df["weight_sum"] = weights.sum(axis=1)
    df = df[df["lagged_choice"] != 0]
    df["SRA_diff"] = df["age"] - df["policy_state_value"]
    df = df[df["sex"] == 0]
    df_s = df[(df["SRA_diff"] == -1.25) & (~df["very_long_insured"])]
    df_s = df_s[(df_s["lagged_choice"] == 3) & (df_s["job_offer"] == 1)]
    df_s.iloc[0].loc[["job_offer", "informed", "health"]]
    # breakpoint()
