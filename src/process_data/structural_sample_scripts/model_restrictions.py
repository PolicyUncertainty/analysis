def enforce_model_choice_restriction(df, specs):
    """This function filters the choice data according to the model setup.

    Specifically, it filters out people retire too early, work too long, or come back
    from retirement,

    """
    max_ret_age = specs["max_ret_age"]

    # Filter out people who are retired before min_ret_age
    # df = df[~((df["choice"] == 0) & (df["age"] < min_long_insured_age))]
    # df = df[~((df["lagged_choice"] == 0) & (df["age"] <= min_long_insured_age))]

    # Filter out people who are working after max_ret_age
    df = df[~((df["choice"] != 0) & (df["age"] >= max_ret_age))]
    # Filter out people who have not retirement as lagged choice after max_ret_age
    df = df[~((df["lagged_choice"] != 0) & (df["age"] > max_ret_age))]
    print(
        str(len(df))
        + " left after dropping people who are working after "
        + str(max_ret_age)
        + "."
    )

    # Filter out people who are unemployed after sra
    df = df[~(((df["age"] - df["policy_state_value"]) >= 0) & (df["choice"] == 1))]
    df = df[
        ~(((df["age"] - df["policy_state_value"]) >= 1) & (df["lagged_choice"] == 1))
    ]

    # Filter out part-time men
    df = df[~((df["sex"] == 0) & (df["choice"] == 2))]
    df = df[~((df["sex"] == 0) & (df["lagged_choice"] == 2))]

    print(
        str(len(df))
        + " left after dropping people are unemployed after the sra and men who work part-time"
    )
    return df
