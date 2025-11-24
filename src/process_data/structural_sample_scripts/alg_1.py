def assign_alg_1_claim(df):
    df["alg_1_claim"] = 0
    df["double_lagged_choice"] = df.groupby(["pid"])["choice"].shift(2)
    df["triple_lagged_choice"] = df.groupby(["pid"])["choice"].shift(3)
    # Lagged choice = 1 and double_lagged = 2,3 -> alg 1 claimant
    alg_1_claimant = (df["lagged_choice"] == 1) & (
        df["double_lagged_choice"].isin([2, 3])
    )
    df.loc[alg_1_claimant, "alg_1_claim"] = 1
    # For individuals => 58, this means even 2
    alg_1_claimant_58_plus = (
        (df["age"] >= 58)
        & (df["lagged_choice"] == 1)
        & (df["double_lagged_choice"].isin([2, 3]))
    )
    df.loc[alg_1_claimant_58_plus, "alg_1_claim"] = 2
    # Generate

    # Above fifty nine tripple lagged choice work is 1
    alg_1_claimant_59_plus = (
        (df["age"] >= 59)
        & (df["lagged_choice"] == 1)
        & (df["triple_lagged_choice"].isin([2, 3]))
        & (df["alg_1_claim"] == 0)
    )
    df.loc[alg_1_claimant_59_plus, "alg_1_claim"] = 1
    return df
