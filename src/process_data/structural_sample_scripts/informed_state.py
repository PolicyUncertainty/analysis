def create_informed_state(df):
    # Initialize as invalid state
    df["informed"] = -99

    # Individuals choosing retirement are informed
    df.loc[df["choice"] == 0, "informed"] = 1
    return df
