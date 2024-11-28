import numpy as np
import pandas as pd
from jax import numpy as jnp


def add_informed_process_specs(specs, path_dict):
    # read informed state transition parameters
    df_uninformed_penalties = pd.read_csv(
        path_dict["est_results"] + "uninformed_average_belief.csv", index_col=0
    )
    df_informed_hazard_rate = pd.read_csv(
        path_dict["est_results"] + "uninformed_hazard_rate.csv", index_col=0
    )

    df_predicted_shares = pd.read_csv(
        path_dict["est_results"] + "predicted_shares.csv", index_col=0
    )
    n_edu_types = len(specs["education_labels"])

    informed_hazard_rate = np.zeros(n_edu_types, dtype=float)
    uninformed_penalties = np.zeros(n_edu_types, dtype=float)
    initial_shares = np.zeros(n_edu_types, dtype=float)

    # Get working ages and also number of working ages
    working_ages = np.arange(specs["start_age"], specs["max_ret_age"] + 1)
    n_working_ages = len(working_ages)
    informed_shares = np.zeros((n_working_ages, n_edu_types), dtype=float)
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        uninformed_penalties[edu_val] = (
            df_uninformed_penalties.loc["erp_uninformed_belief", edu_label] / 100
        )
        informed_hazard_rate[edu_val] = df_informed_hazard_rate.loc[
            "hazard_rate", edu_label
        ]
        initial_shares[edu_val] = df_informed_hazard_rate.loc[
            "initial_informed_share", edu_label
        ]
        informed_shares[:, edu_val] = df_predicted_shares.loc[working_ages, edu_label]

    specs["uninformed_early_retirement_penalty"] = jnp.asarray(uninformed_penalties)
    specs["informed_hazard_rate"] = jnp.asarray(informed_hazard_rate)
    specs["initial_informed_shares"] = jnp.asarray(initial_shares)
    specs["informed_shares"] = jnp.asarray(informed_shares)
    return specs
