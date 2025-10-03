import pandas as pd
import numpy as np

def add_new_life_cycle_results(df_base, df_cf, scenatio_indicator, res_df_life_cycle):

    # Create age index
    ages = np.sort(df_base["age"].unique())

    if res_df_life_cycle is None:
        # initialize result dfs
        res_df_life_cycle = pd.DataFrame(dtype=float, index=ages)

    for append, df_scenario in zip(["base", "cf"], [df_base, df_cf]):

        # Generate overall statistics
        res_df_life_cycle[f"working_hours_{scenatio_indicator}_{append}"] = df_scenario.groupby("age")[
            "working_hours"
        ].aggregate("mean")
        res_df_life_cycle[f"savings_{scenatio_indicator}_{append}"] = df_scenario.groupby("age")[
            "savings_dec"
        ].aggregate("mean")
        res_df_life_cycle[f"consumption_{scenatio_indicator}_{append}"] = df_scenario.groupby("age")[
            "consumption"
        ].aggregate("mean")
        res_df_life_cycle[f"income_{scenatio_indicator}_{append}"] = df_scenario.groupby("age")[
            "total_income"
        ].aggregate("mean")
        res_df_life_cycle[f"assets_{scenatio_indicator}_{append}"] = df_scenario.groupby("age")[
            "savings"
        ].aggregate("mean")

        # Generate rates
        res_df_life_cycle[f"savings_rate_{scenatio_indicator}_{append}"] = (
                res_df_life_cycle[f"savings_{scenatio_indicator}_{append}"]
                / res_df_life_cycle[f"income_{scenatio_indicator}_{append}"]
        )
        res_df_life_cycle[f"employment_rate_{scenatio_indicator}_{append}"] = df_scenario.groupby("age")[
            "choice"
        ].apply(lambda x: x.isin([2, 3]).sum() / len(x))
        res_df_life_cycle[f"retirement_rate_{scenatio_indicator}_{append}"] = df_scenario.groupby("age")[
            "choice"
        ].apply(lambda x: x.isin([0]).sum() / len(x))


    # Generate differences
    res_df_life_cycle[f"savings_rate_diff_{scenatio_indicator}"] = (
            res_df_life_cycle[f"savings_rate_{scenatio_indicator}_cf"]
            - res_df_life_cycle[f"savings_rate_{scenatio_indicator}_base"]
    )
    res_df_life_cycle[f"employment_rate_diff_{scenatio_indicator}"] = (
            res_df_life_cycle[f"employment_rate_{scenatio_indicator}_cf"]
            - res_df_life_cycle[f"employment_rate_{scenatio_indicator}_base"]
    )
    res_df_life_cycle[f"retirement_rate_diff_{scenatio_indicator}"] = (
            res_df_life_cycle[f"retirement_rate_{scenatio_indicator}_cf"]
            - res_df_life_cycle[f"retirement_rate_{scenatio_indicator}_base"]
    )
    return res_df_life_cycle