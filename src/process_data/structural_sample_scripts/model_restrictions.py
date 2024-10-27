def enforce_model_choice_restriction(df, specs):
    """This function filters the choice data according to the model setup.

    Specifically, it filters out people retire too early, work too long, or come back
    from retirement,

    """
    max_ret_age = specs["max_ret_age"]
    min_ret_age = specs["min_ret_age"]
    # Filter out people who are retired before min_ret_age
    df = df[~((df["choice"] == 0) & (df["age"] < min_ret_age))]
    df = df[~((df["lagged_choice"] == 0) & (df["age"] <= min_ret_age))]

    # Filter out people who are working after max_ret_age
    df = df[~((df["choice"] != 0) & (df["age"] >= max_ret_age))]
    # Filter out people who have not retirement as lagged choice after max_ret_age
    df = df[~((df["lagged_choice"] != 0) & (df["age"] > max_ret_age))]
    print(
        str(len(df))
        + " left after dropping people who are retired before "
        + str(min_ret_age)
        + " or working after "
        + str(max_ret_age)
        + "."
    )

    # Filter out people who come back from retirement
    df = df[(df["lagged_choice"] != 0) | (df["choice"] == 0)]

    print(str(len(df)) + " left after dropping people who come back from retirement.")
    return df
