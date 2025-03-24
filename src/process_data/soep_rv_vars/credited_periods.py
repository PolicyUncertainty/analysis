

def create_credited_periods(df):
    """ credited periods for pension of 'particularly long insurance period' (besonders langjährig Versicherte)"""
    # credited periods: VSMO - AZ
    # drop all observations with missing VSMO, set all missing AZ / EZ to 0 (nan coded as 999)
    df = df[df["VSMO"] != 999]
    df.loc[df["AZ"] == 999, "AZ"] = 0
    df.loc[df["AUAZ"] == 999, "AUAZ"] = 0
    #df.loc[df["EZ"] == 999, "EZ"] = 0
    df.loc[:, "credited_periods_months"] = df["VSMO"] - df["AZ"] + df["AUAZ"]  #+ df["EZ"]
    df.loc[:, "credited_periods"] = df["credited_periods_months"] / 12
    # add child raising periods (KBZ, in days & no missings) to credited periods
    df.loc[:, "credited_periods"] = df["credited_periods"] + df["KBZ_TAGE"] / 365
    # keep only old age pensions (RTAT == 2)
    df = df[df["RTAT"] == 2]
    print(f"{df.shape[0]} observations left after dropping disability pensions.")
    # delete pensions of miners, civil servants, war victims, farmers, victims of accidents, and pensioners with foreign pensions 
    df = df[
        ~(df["ismp1"] > 0)
        & ~(df["iciv1"] > 0)
        & ~(df["iwar1"] > 0)
        & ~(df["iagr1"] > 0)
        & ~(df["iguv1"] > 0)
        & ~(df["iaus1"] > 0)
        & ~(df["ilib1"] > 0)
    ]
    print(
        f"{df.shape[0]} observations left after dropping miners, civil servants, war victims, farmers, victims of accidents, and pensioners with foreign pensions."
    )
    return df