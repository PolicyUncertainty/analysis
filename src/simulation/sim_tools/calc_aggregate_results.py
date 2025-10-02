import pandas as pd

from model_code.pension_system.pension_wealth import calc_pension_annuity_value


def add_overall_results(result_df, index, pre_name, df_scenario):

    result_df.loc[index, f"{pre_name}_below_sixty_savings"] = below_sixty_savings(
        df_scenario
    )
    result_df.loc[index, f"{pre_name}_ret_age"] = calc_average_retirement_age(
        df_scenario
    )
    result_df.loc[index, f"{pre_name}_sra_at_ret"] = sra_at_retirement(df_scenario)
    result_df.loc[index, f"{pre_name}_working_hours"] = calc_overall_working_hours(
        df_scenario
    )
    result_df.loc[index, f"{pre_name}_working_hours_below_63"] = (
        calc_working_hours_below_63(df_scenario)
    )


def calc_working_hours_below_63(df):
    mask = df["age"] < 63
    return df.loc[mask, "working_hours"].mean()


def calc_consumption_below_63(df):
    mask = df["age"] < 63
    return df.loc[mask, "consumption"].mean()


def calc_overall_working_hours(df):
    return df["working_hours"].mean()


def calc_average_retirement_age(df):
    fresh_retired_mask = (
        (df["choice"] == 0) & (df["lagged_choice"] != 0) & (df["health"] != 3)
    )

    mean_ret_age = df.loc[fresh_retired_mask, "age"].mean()
    return mean_ret_age


def sra_at_retirement(df):
    fresh_retired_mask = (
        (df["choice"] == 0) & (df["lagged_choice"] != 0) & (df["health"] != 3)
    )

    mean_sra = df.loc[fresh_retired_mask, "policy_state_value"].mean()
    return mean_sra


def private_wealth_at_retirement(df):
    fresh_retired_mask = (
        (df["choice"] == 0) & (df["lagged_choice"] != 0) & (df["health"] != 3)
    )

    mean_wealth = df.loc[fresh_retired_mask, "assets_begin_of_period"].mean()
    return mean_wealth


def pension_wealth_at_retirement(df, specs):
    first_time_pension_payment = (
        (df["lagged_choice"] == 0)
        & (df["policy_state_value"] != 29)
        & (df["health"] != 3)
    )
    df_first = df.loc[first_time_pension_payment, :]
    pension_payments = df_first["gross_retirement_income"]
    ret_age = df_first["age"] - 1
    life_exp = specs["life_exp"][df_first["sex"].values, df_first["education"].values]
    payment_years = life_exp - ret_age
    annuity_value = calc_pension_annuity_value(
        pension_payments=pension_payments,
        payment_years=payment_years,
        interest_rate=specs["interest_rate"],
    )
    return annuity_value.mean()


def expected_lifetime_income(df, specs):
    df["disc_income"] = specs["discount_factor"] ** (df["period"]) * df["total_income"]
    return df.groupby("agent")["disc_income"].sum().mean()


def expected_working_lifetime_income(df, specs):
    mask = df["age"] < 63
    df["disc_income"] = specs["discount_factor"] ** (df["period"]) * df["total_income"]
    return df.loc[mask, :].groupby("agent")["disc_income"].sum().mean()


def expected_pension(df):
    mask = df["lagged_choice"] == 0
    return df.loc[mask, "gross_retirement_income"].mean()


def below_sixty_savings(df):
    below_sixty = df["age"] <= 60

    mean_savings = df.loc[below_sixty, "savings_dec"].mean()
    return mean_savings
