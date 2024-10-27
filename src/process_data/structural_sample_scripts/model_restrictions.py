def enforce_model_choice_restriction(merged_data, min_ret_age, max_ret_age):
    """This function filters the choice data according to the model setup.

    Specifically, it filters out people retire too early, work too long, or come back
    from retirement,

    """
    # Filter out people who are retired before min_ret_age
    merged_data = merged_data[
        ~((merged_data["choice"] == 0) & (merged_data["age"] < min_ret_age))
    ]
    merged_data = merged_data[
        ~((merged_data["lagged_choice"] == 0) & (merged_data["age"] <= min_ret_age))
    ]

    # Filter out people who are working after max_ret_age
    merged_data = merged_data[
        ~((merged_data["choice"] != 0) & (merged_data["age"] >= max_ret_age))
    ]
    # Filter out people who have not retirement as lagged choice after max_ret_age
    merged_data = merged_data[
        ~((merged_data["lagged_choice"] != 0) & (merged_data["age"] > max_ret_age))
    ]
    print(
        str(len(merged_data))
        + " left after dropping people who are retired before "
        + str(min_ret_age)
        + " or working after "
        + str(max_ret_age)
        + "."
    )

    # Filter out people who come back from retirement
    merged_data = merged_data[
        (merged_data["lagged_choice"] != 0) | (merged_data["choice"] == 0)
    ]

    print(
        str(len(merged_data))
        + " left after dropping people who come back from retirement."
    )
    return merged_data
