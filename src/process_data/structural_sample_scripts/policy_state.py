import numpy as np
import pandas as pd


def create_SRA_by_gebjahr(df):
    """This function creates the policy state according to the 2007 reform."""
    gebjahr = df["gebjahr"]

    # Default state is 67
    df["SRA"] = 67.0
    # Create masks for everyone born before 1964
    mask1 = (gebjahr <= 1964) & (gebjahr >= 1958)
    mask2 = (gebjahr <= 1958) & (gebjahr >= 1947)
    mask3 = gebjahr < 1947
    df.loc[mask1, "SRA"] = 67 - 2 / 12 * (1964 - gebjahr[mask1])
    df.loc[mask2, "SRA"] = 66 - 1 / 12 * (1958 - gebjahr[mask2])
    df.loc[mask3, "SRA"] = 65
    return df


def modify_policy_state(df, specs):
    """This function rounds the SRA to the closest multiple of the policy
    expectations process grid size.

    min_SRA is set by the assumption of the belief process in the model.

    """
    min_SRA = specs["min_SRA"]
    SRA_grid_size = specs["SRA_grid_size"]
    policy_states = df["SRA"].values - min_SRA
    policy_states = np.around(policy_states / SRA_grid_size).astype(int)
    policy_state_value = min_SRA + policy_states * SRA_grid_size

    df["policy_state"] = policy_states
    df["policy_state_value"] = policy_state_value
    return df


def create_policy_state(df, specs):

    df["org_SRA"] = df["SRA"].copy()

    # Create age difference. Positive values mean, that individuals retired earlier.
    # For all non-fresh retirees, the difference is 0
    df["age_diff"] = df["age"] - df["corrected_age"]
    # Drop pids with age diff larger than 1
    # First read out pids
    pids_to_drop = df[df["age_diff"].abs() > 1].index.get_level_values("pid").unique()
    # Now drop them
    df = df[~df.index.get_level_values("pid").isin(pids_to_drop)].copy()
    print(
        f"{len(df)} observations left after dropping pids with age diff larger than 1"
    )

    # Manipulate the SRA with the age difference to conserve the distance to the SRA for
    # model ages (integers)
    fresh_mask = (df["choice"] == 0) & (df["lagged_choice"] != 0)
    df.loc[fresh_mask, "SRA"] += df.loc[fresh_mask, "age_diff"]

    # Now we will have for some an SRA below min_SRA (=65). For those we
    # manipulate age as well
    mask_invalid_SRA = df["SRA"] < specs["min_SRA"]
    df.loc[mask_invalid_SRA, "SRA"] += 1
    df.loc[mask_invalid_SRA, "age"] += 1

    df = modify_policy_state(df, specs)

    # Now finally check if all entries of policy_state are valid (above 0)
    if (df["policy_state"] < 0).any():
        raise ValueError("Please revisit policy state definition")

    return df
