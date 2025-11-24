import pickle as pkl

import numpy as np
import pandas as pd
from jax import numpy as jnp


def predict_children_by_state(path_dict, specs):
    """Predicts the number of children in the household for each individual conditional
    on state.

    Produces children array of shape (n_sexes, n_education_types, n_has_partner_states,
    n_periods)

    """
    n_periods = specs["end_age"] - specs["start_age"] + 1
    params = pd.read_csv(
        path_dict["first_step_results"] + "nb_children_estimates.csv",
        index_col=[0, 1, 2],
    )
    # populate numpy ndarray which maps state to average number of children
    children = np.zeros((2, specs["n_education_types"], 2, n_periods))
    for sex in [0, 1]:
        for edu in range(specs["n_education_types"]):
            for has_partner in [0, 1]:
                for period in range(n_periods):
                    predicted_nb_children = (
                        params.loc[(sex, edu, has_partner), "const"]
                        + params.loc[(sex, edu, has_partner), "period"] * period
                        + params.loc[(sex, edu, has_partner), "period_sq"] * period**2
                    )
                    children[sex, edu, has_partner, period] = np.maximum(
                        0, predicted_nb_children
                    )
    max_children = np.max(children, axis=-1)
    return children, max_children


def read_in_partner_transition_specs(paths_dict, specs):

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            type_params = pkl.load(
                open(
                    paths_dict["first_step_results"]
                    + f"result_{sex_label}_{edu_label}.pkl",
                    "rb",
                ),
            )
            for param_name in type_params.keys():
                if param_name not in specs:
                    specs[param_name] = np.zeros(
                        (specs["n_sexes"], specs["n_education_types"]), dtype=float
                    )

                specs[param_name][sex_var, edu_var] = type_params[param_name]

    return specs
