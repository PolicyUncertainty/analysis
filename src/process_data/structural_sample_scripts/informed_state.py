def create_informed_state(df):
    # Initialize as invalid state
    df["informed"] = -99

    # # Individuals choosing retirement are informed
    retired_past_63 = (df["choice"] == 0) & (df["age"] >= 63)
    df.loc[retired_past_63, "informed"] = 1
    return df
