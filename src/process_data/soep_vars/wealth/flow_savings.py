import numpy as np
import pandas as pd

def create_flow_savings(df, specs):
    """
    Create flow savings decision variable based on wealth variable.
    Flow savings is defined as the change in wealth between next period and current period, adjusted for increases due to interest rate.
    """
    interest_rate = specs["interest_rate"]
    df = df.sort_values(by=["pid", "syear"]).copy()

    df["lead_wealth"] = df.groupby("pid")["wealth"].shift(-1, fill_value=np.nan)
    df["savings_dec"] = df["lead_wealth"] / (1 + interest_rate) - df["wealth"]
    df.drop(columns=["lead_wealth"], inplace=True)
    
    return df