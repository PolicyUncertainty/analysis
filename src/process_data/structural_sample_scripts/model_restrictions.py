def _print_filter(before, after, msg):
    pct = (after - before) / before * 100 if before > 0 else 0
    print(f"{after} {msg} ({pct:+.2f}%)")


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
    before1 = len(df)
    df = df[~((df["choice"] != 0) & (df["age"] >= max_ret_age))]
    # Filter out people who have not retirement as lagged choice after max_ret_age
    df = df[~((df["lagged_choice"] != 0) & (df["age"] > max_ret_age))]
    _print_filter(
        before1,
        len(df),
        f"left after dropping people who are working after {max_ret_age}",
    )

    # Filter out people who are unemployed after sra
    before2 = len(df)
    df = df[~(((df["age"] - df["policy_state_value"]) >= 0) & (df["choice"] == 1))]
    df = df[
        ~(((df["age"] - df["policy_state_value"]) >= 1) & (df["lagged_choice"] == 1))
    ]

    # Filter out part-time men
    df = df[~((df["sex"] == 0) & (df["choice"] == 2))]
    df = df[~((df["sex"] == 0) & (df["lagged_choice"] == 2))]
    before3 = len(df)

    _print_filter(
        before2,
        before3,
        "left after dropping people are unemployed after the sra and men who work part-time",
    )
    print(
        str(len(df))
        + " left after dropping people are unemployed after the sra and men who work part-time"
    )
    # # Set job offer for all fresh retirees, which contract could have run out to -99.
    # fresh_retired_mask = (df["choice"] == 0) & (df["lagged_choice"] != 0)
    SRA_diff = df["age"] - df["policy_state_value"]
    # just_after_SRA = (SRA_diff >= 0) & (SRA_diff < 1)
    # df.loc[fresh_retired_mask & just_after_SRA, "job_offer"] = -99
    # Set all people to job_offer zero after sra
    after_SRA = SRA_diff >= 0
    df.loc[after_SRA, "job_offer"] = 0
    df.loc[after_SRA, "choice"] = 0
    one_after_SRA = SRA_diff >= 1
    df.loc[one_after_SRA, "lagged_choice"] = 0
    return df
