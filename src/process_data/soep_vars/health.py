import numpy as np

def create_health_var(data):

    """This function creates the health variables in the soep-PEQUIV dataset.

    - m11126: Self-Rated Health Status
    - m11124: Disability Status of Individual 

    The function replaces the following values in the health variables:
    - [-1] keine Angabe
    - [-2] trifft nicht zu
    - [-5] in Fragebogenversion nicht enthalten

    with np.nan and converts the variables to float.

    The function uses a two category split of the population, encoding 1 if an
    individual has good health and 0 if an individual has bad health. 

    """

    data = data[data["m11126"] >= 0] 
    print(str(len(data)) + " observations left after dropping people with missing health data.")
    
    data = data[data["m11124"] >= 0] 
    print(str(len(data)) + " observations left after dropping people with missing disability data.")

   
    # create health state = 0 if bad health, 1 if good health
    data["health_state"] = 0
    data.loc[data["m11126"].isin([1, 2, 3]) & data["m11124"].isin([0]), "health_state"] = 1

    return data

def clean_health_create_lagged_state(data):
    """
    This function creates a lagged health state variable in the soep-PEQUIV dataset.
    The function replaces the health_state variable with 1 if both the previous and next health_state are 1.
    """

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
    data = data[data["health_state"].notna()] # need to do this again here because spanning the dataframe creates new missing values
    print(str(len(data)) + " observations left after dropping people with missing lagged health data.")

    return data
