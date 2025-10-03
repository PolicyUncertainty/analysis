import numpy as np


def create_health_var(data, filter_missings=True):
    """This function creates the health variables in the soep-PEQUIV dataset.

    - m11126: Self-Rated Health Status
    - m11124: Disability Status of Individual

    The function replaces the following values in the health variables:
    - [-1] keine Angabe
    - [-2] trifft nicht zu
    - [-5] in Fragebogenversion nicht enthalten

    with np.nan and converts the variables to float.

    The function uses a two category split of the population, encoding 0 if an
    individual has good health and 1 if an individual has bad health.

    """
    if filter_missings:
        data = data[data["m11126"] >= 0]
        print(
            str(len(data))
            + " observations left after dropping people with missing health data."
        )

        data = data[data["m11124"] >= 0]
        print(
            str(len(data))
            + " observations left after dropping people with missing disability data."
        )

    # Initialize variable with nan
    data["health"] = np.nan
    # Then assign all valid observations bad health initially (this is all if missings are filtered)
    valid_data = (data["m11126"] >= 0) & (data["m11124"] >= 0)
    data.loc[valid_data, "health"] = 1
    # Then we assign good health to all observations with self assessed good health and no disability
    data.loc[data["m11126"].isin([1, 2, 3]) & data["m11124"].isin([0]), "health"] = 0

    return data


def correct_health_state(data, filter_missings=False):
    """This function creates a lagged health state variable in the soep-PEQUIV dataset.

    The function replaces the health variable with 1 if both the previous and next
    health are 1.

    """

    # replace health with 1 if both previous and next health are 1
    data["lagged_health"] = data.groupby(["pid"])["health"].shift(1)
    data["lead_health"] = data.groupby(["pid"])["health"].shift(-1)

    # one year bad health in between two years of good health is still considered good health
    data.loc[
        (data["lagged_health"] == 0) & (data["lead_health"] == 0),
        "health",
    ] = 0

    # update lead_health
    data["lead_health"] = data.groupby(["pid"])["health"].shift(-1)
    # update lag_health
    data["lagged_health"] = data.groupby(["pid"])["health"].shift(1)

    if filter_missings:
        data = data[data["health"].notna()]
        print(
            str(len(data))
            + " observations left after dropping people with missing health data."
        )

    # drop people with missing lead health data
    # print(str(len(data)) + " observations after spanning the dataframe before dropping people with missing health data.")
    # data = data[data["lead_health"].notna()] # need to do this here because spanning the dataframe creates new missing values
    # print(str(len(data)) + " observations left after dropping people with missing lead health data.")
    # data = data[data["lag_health"].notna()] # need to do this here because spanning the dataframe creates new missing values
    # print(str(len(data)) + " observations left after dropping people with missing lagged health data.")
    # data = data[data["health"].notna()] # need to do this again here because spanning the dataframe creates new missing values
    # print(str(len(data)) + " observations left after dropping people with missing health data.")

    return data
