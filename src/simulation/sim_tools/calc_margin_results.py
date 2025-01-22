def calc_average_retirement_age(df):
    fresh_retired_mask = (df["choice"] == 0) & (df["lagged_choice"] != 0)

    mean_ret_age = df.loc[fresh_retired_mask, "age"].mean()
    return mean_ret_age


def sra_at_retirement(df):
    fresh_retired_mask = (df["choice"] == 0) & (df["lagged_choice"] != 0)

    mean_sra = df.loc[fresh_retired_mask, "policy_state_value"].mean()
    return mean_sra


def below_sixty_savings(df):
    below_sixty = df["age"] <= 60

    mean_savings = df.loc[below_sixty, "savings_dec"].mean()
    return mean_savings
