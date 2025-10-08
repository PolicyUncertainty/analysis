def modify_health_for_disability_pension(df, specs):
    """Modify health variable to incorporate disability pension possibility"""

    # First set age of fresh retirees with age below min_long_insured_age to
    # disabled(2)
    fresh_mask = (df["choice"] == 0) & (df["lagged_choice"] != 0)
    surely_em = fresh_mask & (df["age"] < specs["min_long_insured_age"])
    df.loc[surely_em, "health"] = 2

    # Second, we assume that we do not observe bad health states,
    # i.e. all with bad health are unobserved (-99)
    bad_mask = df["health"] == 1
    df.loc[bad_mask, "health"] = -99

    return df
