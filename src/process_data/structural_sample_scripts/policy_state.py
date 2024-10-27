import numpy as np
import pandas as pd


def create_policy_state(df, specs):
    df["policy_state"] = assign_policy_state_by_gebjahr(df["gebjahr"])

    (
        df["policy_state_value"],
        df["policy_state"],
    ) = modify_policy_state(df["policy_state"], specs)
    return df


def assign_policy_state_by_gebjahr(gebjahr):
    """This function creates the policy state according to the 2007 reform."""
    # Default state is 67
    policy_state = pd.Series(index=gebjahr.index, data=67, dtype=float)
    # Create masks for everyone born before 1964
    mask1 = (gebjahr <= 1964) & (gebjahr >= 1958)
    mask2 = (gebjahr <= 1958) & (gebjahr >= 1947)
    mask3 = gebjahr < 1947
    policy_state.loc[mask1] = 67 - 2 / 12 * (1964 - gebjahr[mask1])
    policy_state.loc[mask2] = 66 - 1 / 12 * (1958 - gebjahr[mask2])
    policy_state.loc[mask3] = 65
    return policy_state


def modify_policy_state(policy_states, specs):
    """This function rounds policy state to the closest multiple of the policy
    expectations process grid size.

    min_SRA is set by the assumption of the belief process in the model.

    """
    min_SRA = specs["min_SRA"]
    SRA_grid_size = specs["SRA_grid_size"]
    policy_states = policy_states - min_SRA
    policy_id = np.around(policy_states / SRA_grid_size).astype(int)
    policy_states_values = min_SRA + policy_id * SRA_grid_size
    return policy_states_values, policy_id
