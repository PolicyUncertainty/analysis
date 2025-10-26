from model_code.pension_system.early_retirement_paths import check_very_long_insured
from specs.experience_pp_specs import add_very_long_insured_specs


def add_very_long_insured_classification(df, path_dict, specs):
    """
    Add a column to the DataFrame indicating whether the individual is classified as
    'very long insured' based on their retirement age and experience years.
    """
    not_retired_mask = df["lagged_choice"] != 0

    df["very_long_insured"] = False

    df_fresh = df.loc[not_retired_mask].copy()
    retirement_age_difference = df_fresh["policy_state_value"] - df_fresh["age"]

    specs = add_very_long_insured_specs(specs, path_dict)

    df.loc[not_retired_mask, "very_long_insured"] = check_very_long_insured(
        retirement_age_difference=retirement_age_difference.values,
        experience_years=df_fresh["experience"].values,
        sex=df_fresh["sex"].values.astype(int),
        model_specs=specs,
    )
    return df
