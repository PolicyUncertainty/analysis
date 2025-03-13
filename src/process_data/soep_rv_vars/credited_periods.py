

def create_credited_periods(df):
    """ credited periods for pension of especially long working life (besonders langjährig Versicherte)"""
    # credited periods: VSMO - AZ
    # drop all observations with missing VSMO, set all missing AZ to 0 (nan coded as 999)
    df = df[df["VSMO"] != 999]
    df.loc[df["AZ"] == 999, "AZ"] = 0
    df["credited_periods_months"] = df["VSMO"] - df["AZ"]
    df["credited_periods"] = df["credited_periods_months"] / 12
    return df