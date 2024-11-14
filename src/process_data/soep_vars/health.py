import numpy as np
def clean_health(data):
    """This function cleans the health variables in the soep-PEQUIV dataset.

    - m11126: Self-Rated Health Status
    - m11124: Disability Status of Individual 

    The function replaces the following values in the health variables:
    - [-1] keine Angabe
    - [-2] trifft nicht zu
    - [-5] in Fragebogenversion nicht enthalten

    with np.nan and converts the variables to float.


    """
    data.rename(columns={"m11126": "srh", "m11124": "disabil" }, inplace=True)
    data.loc[data["srh"] <= -1, "srh"] = np.nan
    data.loc[data["disabil"] <= -1, "disabil"] = np.nan


    # convert to float
    data["srh"] = data["srh"].astype(float)
    data["disabil"] = data["disabil"].astype(float)

    return data

def create_health_states(data):
    """This function creates a health from cleaned versions of m11124 recoded srh (Self-Rated Health Status) and m11124 recoded disabil (Disability Status of Individual) in soep-PEQUIV.

    The function uses a two category split of the population, encoding 1 if an
    individual has good health and 0 if an individual has bad health. 

    """

    # create health state = 0 if bad health, 1 if good health
    data["health_state"] = 0
    data.loc[data["srh"].isin([1, 2, 3]) & data["disabil"].isin([0]), "health_state"] = 1

    # replace health_state with 1 if both previous and next health_state are 1
    data["lagged_health_state"] = data.groupby(["pid"])["health_state"].shift(1)
    data["health_state_next"] = data.groupby(["pid"])["health_state"].shift(-1)
    data.loc[
        (data["health_state"] == 0) & 
        (data["lagged_health_state"] == 1) & 
        (data["health_state_next"] == 1),
        "health_state"
    ] = 1
    
    # drop unnecessary columns
    data.drop(["health_state_next"], axis=1, inplace=True)

    data = data[data["lagged_health_state"].notna()]
    data = data[data["health_state"].notna()]

    return data
