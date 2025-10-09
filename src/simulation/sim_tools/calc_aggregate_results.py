import numpy as np
import pandas as pd

from model_code.pension_system.pension_wealth import calc_pension_annuity_value


def add_overall_results(result_df, index, pre_name, df_scenario, specs):
    """
    Add aggregate results to result_df for a given scenario.

    Parameters:
    -----------
    result_df : pd.DataFrame
        DataFrame to store results
    index : int
        Row index to store results
    pre_name : str
        Prefix for column names (e.g., 'baseline', 'cf')
    df_scenario : pd.DataFrame
        Simulation dataframe for the scenario
    specs : dict, optional
        Model specifications (required for pension wealth calculation)
    """

    # Work Life (<63) metrics
    result_df.loc[index, f"{pre_name}_working_hours_below_63"] = (
        calc_working_hours_below_63(df_scenario)
    )
    result_df.loc[index, f"{pre_name}_consumption_below_63"] = (
        calc_consumption_below_63(df_scenario)
    )
    result_df.loc[index, f"{pre_name}_savings_below_63"] = calc_savings_below_63(
        df_scenario
    )

    # Retirement metrics
    result_df.loc[index, f"{pre_name}_ret_age"] = calc_average_retirement_age(
        df_scenario
    )
    result_df.loc[index, f"{pre_name}_ret_age_excl_disabled"] = (
        calc_average_retirement_age_excl_disabled(df_scenario)
    )
    result_df.loc[index, f"{pre_name}_private_wealth_at_ret"] = (
        private_wealth_at_retirement(df_scenario)
    )
    result_df.loc[index, f"{pre_name}_private_wealth_at_ret_excl_disability"] = (
        private_wealth_excl_disability(df_scenario)
    )

    result_df.loc[index, f"{pre_name}_pension_wealth_at_ret"] = (
        pension_wealth_at_retirement(df_scenario, specs)
    )

    result_df.loc[index, f"{pre_name}_pension_wealth_at_ret_excl_disability"] = (
        pension_wealth_excl_disability(df_scenario, specs)
    )

    result_df.loc[index, f"{pre_name}_pensions"] = pensions(df_scenario)
    result_df.loc[index, f"{pre_name}_pensions_excl_disability"] = (
        pensions_excl_disability(df_scenario)
    )
    result_df.loc[index, f"{pre_name}_share_disability_pensions"] = (
        share_disability_pensions(df_scenario)
    )

    result_df.loc[index, f"{pre_name}_pensions_share_below_63"] = (
        share_disability_pensions_below_63(df_scenario)
    )

    # Lifecycle (30+) metrics
    result_df.loc[index, f"{pre_name}_lifecycle_working_hours"] = (
        calc_lifecycle_working_hours(df_scenario)
    )
    result_df.loc[index, f"{pre_name}_lifecycle_avg_wealth"] = (
        calc_lifecycle_avg_wealth(df_scenario)
    )

    # Keep some existing metrics that might be used elsewhere
    result_df.loc[index, f"{pre_name}_below_sixty_savings"] = below_sixty_savings(
        df_scenario
    )
    result_df.loc[index, f"{pre_name}_sra_at_63"] = sra_at_63(df_scenario)
    result_df.loc[index, f"{pre_name}_sra_at_ret"] = sra_at_retirement(df_scenario)
    result_df.loc[index, f"{pre_name}_working_hours"] = calc_overall_working_hours(
        df_scenario
    )

    return result_df


# ============================================================================
# Work Life (<63) functions
# ============================================================================


def calc_working_hours_below_63(df):
    """Calculate mean annual working hours for age < 63"""
    mask = df["age"] < 63
    return df.loc[mask, "working_hours"].mean()


def calc_consumption_below_63(df):
    """Calculate mean annual consumption for age < 63"""
    mask = df["age"] < 63
    return df.loc[mask, "consumption"].mean() * 10


def calc_savings_below_63(df):
    """Calculate mean annual savings for age < 63"""
    mask = df["age"] < 63
    return df.loc[mask, "savings_dec"].mean() * 10


# ============================================================================
# Retirement functions
# ============================================================================


def sra_at_63(df):
    """Calculate mean SRA at age 63"""
    mask = (df["age"] == 63) & (df["policy_state"] != 29)
    return df.loc[mask, "policy_state_value"].mean()


def calc_average_retirement_age(df):
    """Calculate average pension claiming age"""
    fresh_retired_mask = (
        (df["choice"] == 0) & (df["lagged_choice"] != 0) & (df["health"] != 3)
    )
    mean_ret_age = df.loc[fresh_retired_mask, "age"].mean()
    return mean_ret_age


def calc_average_retirement_age_excl_disabled(df):
    """Calculate average pension claiming age excluding disability pensions"""
    fresh_retired_mask = (
        (df["choice"] == 0) & (df["lagged_choice"] != 0) & (df["health"] != 3)
    )
    non_disabled_mask = df["health"] != 2
    combined_mask = fresh_retired_mask & non_disabled_mask
    mean_ret_age = df.loc[combined_mask, "age"].mean()
    return mean_ret_age


def private_wealth_at_retirement(df):
    """Calculate mean private/financial wealth at retirement"""
    fresh_retired_mask = (
        (df["choice"] == 0) & (df["lagged_choice"] != 0) & (df["health"] != 3)
    )
    mean_wealth = df.loc[fresh_retired_mask, "assets_begin_of_period"].mean()
    return mean_wealth * 10


def private_wealth_excl_disability(df):
    """Calculate mean private/financial wealth at retirement excluding disability pensions"""
    fresh_retired_mask = (
        (df["choice"] == 0) & (df["lagged_choice"] != 0) & (df["health"] != 3)
    )
    non_disabled_mask = df["health"] != 2
    combined_mask = fresh_retired_mask & non_disabled_mask
    mean_wealth = df.loc[combined_mask, "assets_begin_of_period"].mean()
    return mean_wealth * 10


def pension_wealth_at_retirement(df, specs):
    """Calculate mean pension wealth (PV) at retirement"""
    first_time_pension_payment = (
        (df["lagged_choice"] == 0) & (df["policy_state"] != 29) & (df["health"] != 3)
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
    return annuity_value.mean() * 10


def pension_wealth_excl_disability(df, specs):
    """Calculate mean pension wealth (PV) at retirement without disability pensions."""
    first_time_pension_payment = (
        (df["lagged_choice"] == 0)
        & (df["policy_state"] != 29)
        & (~df["health"].isin([2, 3]))
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
    return annuity_value.mean() * 10


def pensions_excl_disability(df):
    """Calculate mean pension at retirement without disability pensions."""
    first_time_pension_payment = (
        (df["lagged_choice"] == 0)
        & (df["policy_state"] != 29)
        & (~df["health"].isin([2, 3]))
    )
    return df.loc[first_time_pension_payment, "gross_retirement_income"].mean() * 10


def pensions(df):
    """Calculate mean pension at retirement."""
    first_time_pension_payment = (
        (df["lagged_choice"] == 0) & (df["policy_state"] != 29) & (df["health"] != 3)
    )
    return df.loc[first_time_pension_payment, "gross_retirement_income"].mean() * 10


def share_disability_pensions(df):
    """Calculate share of disability pensions at retirement."""
    first_time_pension_payment = (
        (df["lagged_choice"] == 0) & (df["policy_state"] != 29) & (df["health"] != 3)
    )
    disability_mask = df["health"] == 2
    n_pensions = np.sum(first_time_pension_payment & disability_mask)
    total_pensions = np.sum(first_time_pension_payment)
    return n_pensions / total_pensions


def share_disability_pensions_below_63(df):
    """Calculate share of fresh retired before 63."""
    first_time_pension_payment = (
        (df["lagged_choice"] != 0) & (df["choice"] == 0) & (df["health"] != 3)
    )
    below_63_mask = df["age"] < 63
    n_pensions_below = np.sum(first_time_pension_payment & below_63_mask)
    total_pensions = np.sum(first_time_pension_payment)
    return n_pensions_below / total_pensions


# ============================================================================
# Lifecycle (30+) functions
# ============================================================================


def calc_lifecycle_working_hours(df):
    """Calculate mean annual working hours for age >= 30"""
    mask = df["age"] >= 30
    return df.loc[mask, "working_hours"].mean()


def calc_lifecycle_avg_wealth(df):
    """Calculate mean financial wealth for age >= 30"""
    mask = df["age"] >= 30
    return df.loc[mask, "savings"].mean() * 10


# ============================================================================
# Legacy/other functions (kept for backward compatibility)
# ============================================================================


def calc_overall_working_hours(df):
    """Calculate mean working hours across all ages"""
    return df["working_hours"].mean()


def sra_at_retirement(df):
    """Calculate mean SRA at retirement"""
    fresh_retired_mask = (
        (df["choice"] == 0) & (df["lagged_choice"] != 0) & (df["health"] != 3)
    )
    mean_sra = df.loc[fresh_retired_mask, "policy_state_value"].mean()
    return mean_sra


def below_sixty_savings(df):
    """Calculate mean savings for age <= 60 (legacy function)"""
    below_sixty = df["age"] <= 60
    mean_savings = df.loc[below_sixty, "savings_dec"].mean()
    return mean_savings * 10


def expected_lifetime_income(df, specs):
    """Calculate expected discounted lifetime income"""
    df["disc_income"] = specs["discount_factor"] ** (df["period"]) * df["total_income"]
    return df.groupby("agent")["disc_income"].sum().mean() * 10


def expected_working_lifetime_income(df, specs):
    """Calculate expected discounted working lifetime income (age < 63)"""
    mask = df["age"] < 63
    df["disc_income"] = specs["discount_factor"] ** (df["period"]) * df["total_income"]
    return df.loc[mask, :].groupby("agent")["disc_income"].sum().mean() * 10


def expected_pension(df):
    """Calculate mean pension payment"""
    mask = df["lagged_choice"] == 0
    return df.loc[mask, "gross_retirement_income"].mean() * 10
