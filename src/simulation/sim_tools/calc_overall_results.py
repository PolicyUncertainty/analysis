import pandas as pd
def calc_overall_results(df_base, df_cf):

    out_row = pd.Series()
    for pre, df_scenario in zip(["cf", "base"], [df_cf, df_base]):
        out_row[f"{pre}_below_sixty_savings"] = below_sixty_savings(df_scenario)
        out_row[f"{pre}_ret_age"] = calc_average_retirement_age(df_scenario)
        out_row[f"{pre}_sra_at_ret"] = sra_at_retirement(df_scenario)
        out_row[f"{pre}_working_hours"] = df_scenario["working_hours"].mean()

    return out_row


def calc_average_retirement_age(df):
    fresh_retired_mask = (
        (df["choice"] == 0) & (df["lagged_choice"] != 0) & (df["health"] != 2)
    )

    mean_ret_age = df.loc[fresh_retired_mask, "age"].mean()
    return mean_ret_age


def sra_at_retirement(df):
    fresh_retired_mask = (
        (df["choice"] == 0) & (df["lagged_choice"] != 0) & (df["health"] != 2)
    )

    mean_sra = df.loc[fresh_retired_mask, "policy_state_value"].mean()
    return mean_sra


def below_sixty_savings(df):
    below_sixty = df["age"] <= 60

    mean_savings = df.loc[below_sixty, "savings_dec"].mean()
    return mean_savings

