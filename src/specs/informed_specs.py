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
    ages_until_max_ret = np.arange(0, specs["max_ret_age"] + 1)
    informed_shares_in_ages = np.zeros(
        (len(ages_until_max_ret), n_edu_types), dtype=float
    )
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
        predicted_ages = df_predicted_shares.index.values
        informed_shares_in_ages[predicted_ages, edu_val] = df_predicted_shares.loc[
            predicted_ages, edu_label
        ]

    informed_shares_in_ages[:, :] = 1.0

    specs["uninformed_ERP"] = jnp.asarray(uninformed_penalties)
    specs["informed_hazard_rate"] = jnp.asarray(informed_hazard_rate)
    specs["informed_shares_in_ages"] = jnp.asarray(informed_shares_in_ages)
    return specs
