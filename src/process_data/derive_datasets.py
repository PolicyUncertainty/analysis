import os
import pandas as pd


def gather_decision_data(paths, df = None, load_data=False):
    
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "decision_data.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data
    elif df is None:
        raise ValueError("Either set load_data=True or provide df as input.")

    # Keep relevant columns (i.e. state variables)
    df = df[
        [
            "choice",
            "period",
            "lagged_choice",
            "policy_state",
            "policy_state_value",
            "retirement_age_id",
            "experience",
            "wealth",
        ]
    ]
    df = df.astype(
        {
            "choice": "int8",
            "lagged_choice": "int8",
            "policy_state": "int8",
            "retirement_age_id": "int8",
            "experience": "int8",
            "wealth": "float32",
            "period": "int8",
        }
    )

    print(str(len(df)) + " observations in final decision data.")

    # Anonymize and save data
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(out_file_path)
    return df

def gather_wage_data(paths, df=None, load_data=False):
    if not os.path.exists(paths["intermediate_data"]):
        os.makedirs(paths["intermediate_data"])

    out_file_path = paths["intermediate_data"] + "wage_data.pkl"

    if load_data:
        data = pd.read_pickle(out_file_path)
        return data
    elif df is None:
        raise ValueError("Either set load_data=True or provide df as input.")

    # Turn index (pid, syear) into columns
    df.reset_index(inplace=True)

    # Keep relevant columns for fixed effects estimation
    df = df[
        [
            "wage",
            "experience",
            "pid",
            "syear",
        ]
    ]
    df = df.astype(
        {
            "wage": "float32",
            "experience": "int16",
            "pid": "int32",
            "syear": "int16",
        }
    )

    print(str(len(df)) + " observations in final wage data.")

    # Save data
    df.to_pickle(out_file_path)
    return df